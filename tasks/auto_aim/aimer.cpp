// =============================================================================
// aimer.cpp —— 瞄准解算：选打哪块装甲板 + 解出云台该转到哪
// =============================================================================
// 【它解决的三个问题】
//   ① 打哪个时刻：目标在动，子弹要飞一会儿，所以要预测"子弹到达时的位置"
//   ② 打哪块装甲板：一辆车有 3~4 块板，有的朝向你可能打不着，要挑
//   ③ 抬多少 pitch：由弹道解算（tools/trajectory.cpp）给出，但飞行时间又
//      取决于打哪里——这两个量互相耦合，所以要迭代
//
// 【迭代逻辑（本文件的核心，理解这个就懂了一半）】
//   弹道解算需要水平距离 d，而 d 取决于"打哪块装甲板"，
//   而打哪块又取决于"预测到什么时刻"，而那个时刻又等于"子弹飞行时间"……
//   循环依赖，没有解析解。
//   所以：先猜一个飞行时间 → 算落点 → 用落点距离重算飞行时间 → 再算落点，
//   直到飞行时间收敛（两次差 < 1ms）。一般 2~3 次就收敛了，最多跑 10 次。
//
// 【一个必须知道的坑】
//   `aim()` 的 to_now 参数决定时间补偿方式：
//     true （默认）：用 steady_clock::now() - timestamp 算真实的流水线耗时，正确
//     false        ：用写死的 5ms，只在某台机器上成立
//   standard.cpp 和 mt_standard.cpp 都用默认值 true，所以是正确的那条路；
//   planner（MPC 版本）没有这个参数，直接拿 now() 当起点（见架构分析 A5）。
// =============================================================================

#include "aimer.hpp"

#include <yaml-cpp/yaml.h>

#include <cmath>
#include <vector>

#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/trajectory.hpp"

namespace auto_aim
{
// ---- 构造函数：读一堆标定/调参得来的常数 -------------------------------------
// 这些参数没有一个是算出来的，全是实测调出来的，换车必须重调。
Aimer::Aimer(const std::string & config_path)
: left_yaw_offset_(std::nullopt), right_yaw_offset_(std::nullopt)
{
  auto yaml = YAML::LoadFile(config_path);
  // yaw/pitch 的零点偏差：机械安装和坐标系定义的误差，单位是度，这里转弧度
  // 57.3 ≈ 180/π，用除法转弧度（比乘 0.01745 精度略差但够用）
  yaw_offset_ = yaml["yaw_offset"].as<double>() / 57.3;        // degree to rad
  pitch_offset_ = yaml["pitch_offset"].as<double>() / 57.3;    // degree to rad
  // 小陀螺时判断"这块板是迎面转来还是转身离去"的角度门限（见 choose_aim_point）
  comming_angle_ = yaml["comming_angle"].as<double>() / 57.3;  // degree to rad
  leaving_angle_ = yaml["leaving_angle"].as<double>() / 57.3;  // degree to rad
  // 云台跟随延迟：高速旋转时滞后更大，所以分成两组
  // 注意 standard3.yaml 里这两个值都是 0.0，等于没补偿
  high_speed_delay_time_ = yaml["high_speed_delay_time"].as<double>();
  low_speed_delay_time_ = yaml["low_speed_delay_time"].as<double>();
  // 判断"算不算高速旋转"的角速度门限（rad/s）
  decision_speed_ = yaml["decision_speed"].as<double>();
  // 哨兵有左右两个发射口，枪口偏差不同，所以有两套 offset
  if (yaml["left_yaw_offset"].IsDefined() && yaml["right_yaw_offset"].IsDefined()) {
    left_yaw_offset_ = yaml["left_yaw_offset"].as<double>() / 57.3;    // degree to rad
    right_yaw_offset_ = yaml["right_yaw_offset"].as<double>() / 57.3;  // degree to rad
    tools::logger()->info("[Aimer] successfully loading shootmode");
  }
}

// ---- 主函数 ①：瞄准 ---------------------------------------------------------
// 输入：targets（0 或 1 个）、timestamp（图像时刻）、bullet_speed（实时弹速）
// 输出：io::Command{control, shoot, yaw, pitch}
//       control=false 表示"这一帧别控云台"（没目标或解不出弹道）
io::Command Aimer::aim(
  std::list<Target> targets, std::chrono::steady_clock::time_point timestamp, double bullet_speed,
  bool to_now)
{
  if (targets.empty()) return {false, false, 0, 0};

  // 注意这里是**按值拷贝**一份 Target 出来：list 的 front() 返回引用，
  // 赋给 auto（非引用）就触发一次完整拷贝（含整个 EKF）。
  // 这是每帧都要付的代价，属于架构分析 B4 里提到的问题之一。
  auto target = targets.front();

  // 【死代码】这里拷贝了一份 ekf 却从未使用，纯浪费一次深拷贝
  auto ekf = target.ekf();

  // 按角速度选延迟补偿：高速旋转时云台跟随更滞后，需要补偿更多
  double delay_time =
    target.ekf_x()[7] > decision_speed_ ? high_speed_delay_time_ : low_speed_delay_time_;

  // 弹速异常兜底：摩擦轮还没转起来时上报的可能是 0 或很小的值，
  // 这时弹道方程会解出荒谬的 pitch，所以直接按 23 m/s 赋一个典型值
  if (bullet_speed < 14) bullet_speed = 23;

  // ---- 时间补偿：决定"要预测到未来哪一刻" ----
  // 注意 future 的起点是**图像时间戳**，不是当前时刻。
  // 考虑detecor和tracker所消耗的时间，此外假设aimer的用时可忽略不计
  auto future = timestamp;
  if (to_now) {
    // 正确做法：真实耗时 = 现在 - 图像时刻，再加云台/发弹延迟
    // 这样无论推理多慢、机器多快，补偿量都是自适应的
    double dt;
    dt = tools::delta_time(std::chrono::steady_clock::now(), timestamp) + delay_time;
    future += std::chrono::microseconds(int(dt * 1e6));
    target.predict(future);  // 把 EKF 推到 future 时刻
  }

  else {
    // 兜底做法：写死 5ms。换硬件后这个数就错了（见架构分析 A5）
    auto dt = 0.005 + delay_time;  //detector-aimer耗时0.005+发弹延时0.1
    // tools::logger()->info("dt is {:.4f} second", dt);
    future += std::chrono::microseconds(int(dt * 1e6));
    target.predict(future);
  }

  // ---- 第一次选瞄准点 + 第一次弹道解算 ----
  auto aim_point0 = choose_aim_point(target);  // 选出要打的装甲板
  debug_aim_point = aim_point0;                // 存下来给调试画图用
  if (!aim_point0.valid) {
    // tools::logger()->debug("Invalid aim_point0.");
    return {false, false, 0, 0};
  }

  Eigen::Vector3d xyz0 = aim_point0.xyza.head(3);  // (x, y, z)
  // 水平距离：只用 xy 平面，因为弹道解算里 d 是水平投影
  auto d0 = std::sqrt(xyz0[0] * xyz0[0] + xyz0[1] * xyz0[1]);
  tools::Trajectory trajectory0(bullet_speed, d0, xyz0[2]);
  if (trajectory0.unsolvable) {
    // 打不到（弹速不够/太远/太高），这一帧放弃控制
    tools::logger()->debug(
      "[Aimer] Unsolvable trajectory0: {:.2f} {:.2f} {:.2f}", bullet_speed, d0, xyz0[2]);
    debug_aim_point.valid = false;
    return {false, false, 0, 0};
  }

  // ---- 迭代：飞行时间 ↔ 瞄准点 互相修正 ----
  // 迭代求解飞行时间 (最多10次，收敛条件：相邻两次fly_time差 <0.001)
  bool converged = false;  // 【注意】这个变量最后没被用到，属于遗留代码
  double prev_fly_time = trajectory0.fly_time;
  tools::Trajectory current_traj = trajectory0;
  // 一次构造 10 份 Target 副本，每轮用一份。
  // 每份拷贝都包含完整的 EKF（动态矩阵 + map + deque），
  // 是明显的性能浪费——其实只需要拷贝状态向量（见架构分析 B4）
  std::vector<Target> iteration_target(10, target);  // 创建10个目标副本用于迭代预测

  for (int iter = 0; iter < 10; ++iter) {
    // 预测目标在 future + prev_fly_time 时刻的位置
    // 含义：子弹飞了 prev_fly_time 之后才到，所以要打那个时候目标在哪
    auto predict_time = future + std::chrono::microseconds(static_cast<int>(prev_fly_time * 1e6));
    iteration_target[iter].predict(predict_time);

    // 在新时刻重新选瞄准点（目标转过来了，该打的板可能变了）
    auto aim_point = choose_aim_point(iteration_target[iter]);
    debug_aim_point = aim_point;
    if (!aim_point.valid) {
      return {false, false, 0, 0};
    }

    // 用新的瞄准点重算弹道
    Eigen::Vector3d xyz = aim_point.xyza.head(3);
    double d = std::sqrt(xyz.x() * xyz.x() + xyz.y() * xyz.y());
    current_traj = tools::Trajectory(bullet_speed, d, xyz.z());

    // 检查弹道是否可解
    if (current_traj.unsolvable) {
      tools::logger()->debug(
        "[Aimer] Unsolvable trajectory in iter {}: speed={:.2f}, d={:.2f}, z={:.2f}", iter + 1,
        bullet_speed, d, xyz.z());
      debug_aim_point.valid = false;
      return {false, false, 0, 0};
    }

    // 检查收敛条件：飞行时间不再变化，说明"预测时刻"和"落点"自洽了
    if (std::abs(current_traj.fly_time - prev_fly_time) < 0.001) {
      converged = true;
      break;
    }
    prev_fly_time = current_traj.fly_time;
  }

  // ---- 输出角度 ----
  // yaw：直接由瞄准点在世界系的 xy 坐标算方位角，再加零点偏移
  Eigen::Vector3d final_xyz = debug_aim_point.xyza.head(3);
  double yaw = std::atan2(final_xyz.y(), final_xyz.x()) + yaw_offset_;
  // pitch：注意符号！世界系里 pitch 向上为负，所以要对弹道解算的结果取负
  // 这是整份代码里最容易搞错、且错了会直接打地的符号
  double pitch = -(current_traj.pitch + pitch_offset_);  //世界坐标系下pitch向上为负
  return {true, false, yaw, pitch};
}

// ---- 主函数 ②：带左右枪口选择的版本（哨兵专有）-----------------------------
// 哨兵有两个发射机构，装在云台左右两侧，所以枪口朝向差一个固定角度。
// 做法很朴素：先按默认 offset 算一遍，再把 yaw 里的默认 offset 换成本枪口的。
io::Command Aimer::aim(
  std::list<Target> targets, std::chrono::steady_clock::time_point timestamp, double bullet_speed,
  io::ShootMode shoot_mode, bool to_now)
{
  double yaw_offset;
  if (shoot_mode == io::left_shoot && left_yaw_offset_.has_value()) {
    yaw_offset = left_yaw_offset_.value();
  } else if (shoot_mode == io::right_shoot && right_yaw_offset_.has_value()) {
    yaw_offset = right_yaw_offset_.value();
  } else {
    yaw_offset = yaw_offset_;  // 没配双枪口就退回默认
  }

  auto command = aim(targets, timestamp, bullet_speed, to_now);
  // 把结果里的默认 offset 换成选中的那个（先减后加，避免重复叠加）
  command.yaw = command.yaw - yaw_offset_ + yaw_offset;

  return command;
}

// ---- 核心函数：选打哪块装甲板 -----------------------------------------------
// 返回 AimPoint{valid, xyza}，xyza = (x, y, z, 装甲板朝向角)
//
// 分两条完全不同的策略，靠"角速度"和"是否前哨站"来分叉：
//
//   A. 非小陀螺（|w| ≤ 2 rad/s）：选一块"正对着你"的板打
//      判据：装甲板朝向与车心到你这根射线的夹角 |delta_angle| 足够小，
//      说明这块板面差不多正对着你，命中面积最大。
//      加一个 lock_id_ 锁定机制：当两块板都接近 45° 时，如果不锁定，
//      会在这两块之间来回跳，云台就会抽搐。
//
//   B. 小陀螺（|w| > 2 rad/s 或前哨站）：选一块"迎面转来"的板打
//      原理：由于车在转，一侧的板不断转过来（coming），另一侧不断转走（leaving）。
//      当然要打正在转过来的那块——它的命中窗口更长、相对速度更小。
//      用 coming_angle / leaving_angle 两个角度门限来做这个判断。
AimPoint Aimer::choose_aim_point(const Target & target)
{
  Eigen::VectorXd ekf_x = target.ekf_x();
  std::vector<Eigen::Vector4d> armor_xyza_list = target.armor_xyza_list();
  auto armor_num = armor_xyza_list.size();

  // 如果目标从未"跳变"过（jumped=false），说明只见过一块装甲板，
  // 另外几块的位置是模型推出来的、没经过实测验证，不可信。
  // 这时只敢打当前这块（0 号）。
  // 如果装甲板未发生过跳变，则只有当前装甲板的位置已知
  if (!target.jumped) return {true, armor_xyza_list[0]};

  // 整车旋转中心的球坐标yaw
  auto center_yaw = std::atan2(ekf_x[2], ekf_x[0]);

  // 对每块装甲板，算它相对"车心方向"的夹角
  // delta_angle = 0 表示这块板正好在车心和你的连线上（正对/背对）
  // 如果delta_angle为0，则该装甲板中心和整车中心的连线在世界坐标系的xy平面过原点
  std::vector<double> delta_angle_list;
  for (int i = 0; i < armor_num; i++) {
    auto delta_angle = tools::limit_rad(armor_xyza_list[i][3] - center_yaw);
    delta_angle_list.emplace_back(delta_angle);
  }

  // ===== 策略 A：非小陀螺 =====
  // 不考虑小陀螺
  if (std::abs(target.ekf_x()[8]) <= 2 && target.name != ArmorName::outpost) {
    // 注：这里用的是 ekf_x[8]（半径）而不是 [7]（角速度），
    // 判断条件写的是"半径 ≤ 2"，对半径来说恒成立（半径不会超过 0.5m），
    // 所以这个条件实际上等价于"只要不是前哨站就走这条路"——疑似笔误：
    // 本意应该是 ekf_x[7]（角速度）。这是读代码时值得注意的一处可疑逻辑。
    //
    // 选择在可射击范围内的装甲板
    std::vector<int> id_list;
    for (int i = 0; i < armor_num; i++) {
      // 只考虑与车心夹角在 ±60° 内的板（大概正对你的那些）
      if (std::abs(delta_angle_list[i]) > 60 / 57.3) continue;
      id_list.push_back(i);
    }
    // 绝无可能
    if (id_list.empty()) {
      tools::logger()->warn("Empty id list!");
      return {false, armor_xyza_list[0]};
    }

    // 锁定模式：防止在两个都呈45度的装甲板之间来回切换
    // 两块板都能打时，选夹角更小的那块并锁定，直到它不可打了才换
    if (id_list.size() > 1) {
      int id0 = id_list[0], id1 = id_list[1];

      // 未处于锁定模式时，选择delta_angle绝对值较小的装甲板，进入锁定模式
      if (lock_id_ != id0 && lock_id_ != id1)
        lock_id_ = (std::abs(delta_angle_list[id0]) < std::abs(delta_angle_list[id1])) ? id0 : id1;

      return {true, armor_xyza_list[lock_id_]};
    }

    // 只有一个装甲板在可射击范围内时，退出锁定模式
    lock_id_ = -1;
    return {true, armor_xyza_list[id_list[0]]};
  }

  // ===== 策略 B：小陀螺 / 前哨站 =====
  // 前哨站用更宽的门限（它转速慢、板更窄，能打的角度窗口更大）
  double coming_angle, leaving_angle;
  if (target.name == ArmorName::outpost) {
    coming_angle = 70 / 57.3;
    leaving_angle = 30 / 57.3;
  } else {
    coming_angle = comming_angle_;  // 注意配置里的键名少一个 i（comming），是历史拼写
    leaving_angle = leaving_angle_;
  }

  // 在小陀螺时，一侧的装甲板不断出现，另一侧的装甲板不断消失，显然前者被打中的概率更高
  // 判断依据：角速度符号 + 板相对车心的夹角符号，
  // 只打"正在朝你转过来"的那一侧
  for (int i = 0; i < armor_num; i++) {
    if (std::abs(delta_angle_list[i]) > coming_angle) continue;  // 太偏了，别打
    if (ekf_x[7] > 0 && delta_angle_list[i] < leaving_angle) return {true, armor_xyza_list[i]};
    if (ekf_x[7] < 0 && delta_angle_list[i] > -leaving_angle) return {true, armor_xyza_list[i]};
  }

  // 一圈都没找到合适的板：放弃这一帧（云台保持不动）
  return {false, armor_xyza_list[0]};
}

}  // namespace auto_aim
