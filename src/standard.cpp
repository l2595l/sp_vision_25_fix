// =============================================================================
// standard.cpp —— 单线程自瞄主程序（步兵/英雄）
// =============================================================================
// 【在整个项目里的位置】
//   这是最朴素的一条流水线：采集 → 检测 → 跟踪 → 瞄准 → 下发，全部在同一个
//   线程里顺序执行。它是理解整个 sp_vision 的最佳入口，因为没有任何并发。
//
// 【它组装了哪些模块】（读代码时按这个顺序跟下去）
//   io::Camera         → io/camera.cpp              读图 + 时间戳
//   io::CBoard         → io/cboard.cpp              IMU 四元数 / 下发指令
//   auto_aim::YOLO     → tasks/auto_aim/yolo.cpp    神经网络检测装甲板
//   auto_aim::Solver   → tasks/auto_aim/solver.cpp  像素 → 世界坐标（PnP）
//   auto_aim::Tracker  → tasks/auto_aim/tracker.cpp 多帧关联 + 状态机
//   auto_aim::Aimer    → tasks/auto_aim/aimer.cpp   预测 + 弹道解算
//   auto_aim::Shooter  → tasks/auto_aim/shooter.cpp 开火时机判断
//
// 【为什么只有 80 行】
//   算法全在类里，main 只负责"接线"和"循环"。这也是它能被复制成 13 个不同
//   兵种程序而不出错的原因——差异只在配置文件（yaml）和通信方式（CAN/串口）。
// =============================================================================

#include <fmt/core.h>

#include <chrono>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>

#include "io/camera.hpp"
#include "io/cboard.hpp"
#include "tasks/auto_aim/aimer.hpp"
#include "tasks/auto_aim/multithread/commandgener.hpp"
#include "tasks/auto_aim/shooter.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/tracker.hpp"
#include "tasks/auto_aim/yolo.hpp"
#include "tools/exiter.hpp"
#include "tools/img_tools.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/plotter.hpp"
#include "tools/recorder.hpp"

using namespace std::chrono;

// OpenCV 的命令行参数描述，格式：{名字 简写 帮助 | 默认值 | 说明}
// 这里只定义了一个位置参数 @config-path，默认指向步兵配置
const std::string keys =
  "{help h usage ? |      | 输出命令行参数说明}"
  "{@config-path   | configs/standard3.yaml | 位置参数，yaml配置文件路径 }";

int main(int argc, char * argv[])
{
  // ---------------------------------------------------------------- 参数解析
  cv::CommandLineParser cli(argc, argv, keys);
  auto config_path = cli.get<std::string>(0);  // 0 表示第 0 个位置参数
  if (cli.has("help") || config_path.empty()) {
    cli.printMessage();
    return 0;
  }

  // ------------------------------------------------- 全局服务（与算法无关的工具）
  // Exiter  ：注册 SIGINT/SIGTERM，exit() 返回 true 时让循环优雅退出
  // Plotter ：把调试数据以 JSON over UDP 发到 127.0.0.1:9870，配合脚本画曲线
  // Recorder：把 (图像, 四元数, 时间戳) 落盘，用于事后回放调试（默认关闭）
  tools::Exiter exiter;
  tools::Plotter plotter;
  tools::Recorder recorder;

  // ------------------------------------------------------------ 构造所有模块
  // 构造顺序有讲究：Camera/CBoard 会启动后台线程读硬件，必须先就绪，
  // 否则后面算法线程取数据时会阻塞。
  // 每个模块的构造函数都只做一件事：从同一个 yaml 里读自己关心的几项。
  io::CBoard cboard(config_path);  // 开 SocketCAN，后台线程收 IMU/弹速/模式
  io::Camera camera(config_path);  // 开相机，后台线程持续取帧进队列

  auto_aim::YOLO detector(config_path, false);     // 第二参数 false = 不开 imshow
  auto_aim::Solver solver(config_path);            // 载入相机内参 + 手眼外参
  auto_aim::Tracker tracker(config_path, solver);  // 需要 solver 做坐标变换
  auto_aim::Aimer aimer(config_path);
  auto_aim::Shooter shooter(config_path);

  // 循环外的这几个变量承载"跨帧状态"：图像、姿态、时间戳、模式
  cv::Mat img;
  Eigen::Quaterniond q;
  std::chrono::steady_clock::time_point t;

  auto mode = io::Mode::idle;       // 当前模式：idle / auto_aim / small_buff / big_buff
  auto last_mode = io::Mode::idle;  // 上一轮模式，用于检测切换并打日志

  // ================================================================== 主循环
  while (!exiter.exit()) {
    // ① 取一帧图，同时拿到这一帧的时间戳 t
    //    t 的语义见 io/hikrobot.cpp：是"主机取到帧的时刻"，不是曝光时刻。
    //    这一点直接决定了后面所有时间补偿的上限（详见架构分析 A5）。
    camera.read(img, t);

    // ② 取与该时刻对齐的云台姿态
    //    imu_at() 内部用 slerp 在相邻两帧 IMU 之间插值，且会阻塞等待
    //    timestamp 之后的数据。传 t - 1ms 是个经验补偿，避免最新数据还没到
    //    就卡死在这里。
    q = cboard.imu_at(t - 1ms);
    mode = cboard.mode;  // 模式由下位机通过 CAN 下发

    // ③ 模式变化时打一条日志，方便事后对齐时序
    if (last_mode != mode) {
      tools::logger()->info("Switch to {}", io::MODES[mode]);
      last_mode = mode;
    }

    // ④ 录制（默认注释）。调试时打开，录下的数据可离线回放整条链路
    // recorder.record(img, q, t);

    // ⑤ 更新 gimbal→world 旋转：把 IMU 四元数转成"云台系到世界系"的矩阵
    //    这是后续所有解算的地基，必须每帧先更新
    solver.set_R_gimbal2world(q);

    // ⑥ 当前云台 yaw/pitch/roll（本文件没用到，保留给调试/扩展）
    Eigen::Vector3d ypr = tools::eulers(solver.R_gimbal2world(), 2, 1, 0);

    // ⑦ 检测：BGR 图 → 装甲板列表（四角点、编号、颜色、置信度）
    //    输出仍是像素坐标，还没进入世界坐标系
    auto armors = detector.detect(img);

    // ⑧ 跟踪：把本帧检测结果接到历史轨迹上，输出 Target（内含 EKF 状态）
    //    关键：track() 内部会调用 solver.solve() 完成像素→世界的转换
    auto targets = tracker.track(armors, t);

    // ⑨ 瞄准：预测目标在未来某时刻的位置 → 选瞄准点 → 解弹道 → 得 yaw/pitch
    //    to_now 默认为 true，会用 steady_clock::now() 计算真实流水线耗时
    auto command = aimer.aim(targets, t, cboard.bullet_speed);

    // ⑩ 下发：拼成 8 字节 CAN 帧发给下位机
    //    注意 io::Command 只有 yaw/pitch，没有速度/加速度——这是 MPC 前馈
    //    在这条链路上失效的根因（详见架构分析）
    cboard.send(command);
  }

  return 0;
}
