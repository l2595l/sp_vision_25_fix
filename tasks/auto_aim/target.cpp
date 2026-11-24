// =============================================================================
// target.cpp —— 单目标的状态估计（11 维 EKF）
// =============================================================================
// 【这个文件要解决的问题】
//   图像里每帧只能看到敌人 1~2 块装甲板，而且还会被遮挡、被小陀螺转走。
//   如果直接跟踪"装甲板"，一旦看不见就丢了。
//   所以这里改成跟踪**整车的旋转中心**：中心一直存在，装甲板只是挂在中心周围、
//   按固定角度分布的几个点。只要估计出中心和转速，就能算出任意时刻
//   4 块装甲板分别在哪。
//
// 【11 维状态向量 x 的含义】（这是读懂本文件的关键）
//   x[0]  cx   整车中心 x（世界系，米）
//   x[1]  vx   中心 x 方向速度
//   x[2]  cy   中心 y
//   x[3]  vy
//   x[4]  cz   中心 z（高度）
//   x[5]  vz
//   x[6]  a    当前这块装甲板的 yaw 角（弧度）—— 注意是"装甲板角度"，不是车身角
//   x[7]  w    角速度（小陀螺时就是转速）
//   x[8]  r    中心到装甲板的半径（米）
//   x[9]  l    半径修正量：相隔 90° 的那对装甲板，半径可能是 r + l
//   x[10] h    高度修正量：相隔 90° 的那对装甲板，高度可能是 cz + h
//
//   为什么要有 l 和 h：不是所有车的 4 块装甲板都对称。比如平衡步兵/英雄，
//   前后两块和左右两块装的半径、高度都不同。用 l/h 一个参数描述这种"交错差异"，
//   比给每块装甲板都建一个状态要省得多。
//
// 【观测是什么】
//   一次检测只给一块装甲板的 [yaw, pitch, distance]（球坐标）和它的 [yaw]，
//   也就是 4 维观测 z。要用 4 维观测去更新 11 维状态，中间靠观测方程 h(x) 联系，
//   而 h(x) 的前提是"我知道这块装甲板是第几号"——这就是 update() 里那段
//   匹配逻辑存在的意义。
// =============================================================================

#include "target.hpp"

#include <numeric>

#include "tools/logger.hpp"
#include "tools/math_tools.hpp"

namespace auto_aim
{
// ---- 构造函数 ①：从"一块装甲板的观测"初始化整车状态 -------------------------
// 参数 armor      ：首次检测到的装甲板（已经过 solver.solve，有世界坐标）
// 参数 t          ：该帧时间戳
// 参数 radius     ：该兵种的整车半径（从配置或 tracker 里按兵种给的经验值）
// 参数 armor_num  ：该兵种有几块装甲板（3 或 4）
// 参数 P0_dig     ：初始协方差的对角线，来自 tracker.set_target()，按兵种不同
//
// 关键动作：把"装甲板位置"反推成"整车中心位置"
//   装甲板在世界系的坐标 = 中心 - r · (cos a, sin a)
//   所以要反推中心：center = armor_xyz + r · (cos a, sin a)
//   这里的正负号必须和 h_armor_xyz() 里的方向严格一致，否则滤波器一开始就是错的。
Target::Target(
  const Armor & armor, std::chrono::steady_clock::time_point t, double radius, int armor_num,
  Eigen::VectorXd P0_dig)
: name(armor.name),
  armor_type(armor.type),
  jumped(false),
  last_id(0),
  update_count_(0),
  armor_num_(armor_num),
  t_(t),
  is_switch_(false),
  is_converged_(false),
  switch_count_(0)
{
  auto r = radius;
  priority = armor.priority;
  const Eigen::VectorXd & xyz = armor.xyz_in_world;  // 装甲板位置
  const Eigen::VectorXd & ypr = armor.ypr_in_world;  // 装甲板姿态

  // 旋转中心的坐标（由装甲板位置 + 半径沿 yaw 方向推出）
  auto center_x = xyz[0] + r * std::cos(ypr[0]);
  auto center_y = xyz[1] + r * std::sin(ypr[0]);
  auto center_z = xyz[2];

  // 初始状态：位置取反推值，速度全部为 0（假设刚发现时目标匀速/静止），
  // 角度取观测值，角速度 0，半径取经验值，l/h 修正量 0
  // x vx y vy z vz a w r l h
  // a: angle
  // w: angular velocity
  // l: r2 - r1
  // h: z2 - z1
  Eigen::VectorXd x0{{center_x, 0, center_y, 0, center_z, 0, ypr[0], 0, r, 0, 0}};  //初始化预测量
  Eigen::MatrixXd P0 = P0_dig.asDiagonal();  // 初始不确定度，由调用方按兵种给定

  // 角度是周期量（-π 和 +π 是同一个方向），普通加法会在跨越 ±π 时产生
  // 2π 的跳变。这个自定义加法在每次相加后把角度折回 [-π, π]。
  auto x_add = [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) -> Eigen::VectorXd {
    Eigen::VectorXd c = a + b;
    c[6] = tools::limit_rad(c[6]);
    return c;
  };

  ekf_ = tools::ExtendedKalmanFilter(x0, P0, x_add);  //初始化滤波器（预测量、预测量协方差）
}

// ---- 构造函数 ②：仿真/调试用 ------------------------------------------------
// 直接给定位置、转速、半径、高度构造一个"理想目标"，不依赖任何检测结果。
// P0 全 0 表示"完全确信自己的初值"，用于 tests/planner_test_offline 这类离线验证。
Target::Target(double x, double vyaw, double radius, double h) : armor_num_(4)
{
  Eigen::VectorXd x0{{x, 0, 0, 0, 0, 0, 0, vyaw, radius, 0, h}};
  Eigen::VectorXd P0_dig{{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}};
  Eigen::MatrixXd P0 = P0_dig.asDiagonal();

  auto x_add = [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) -> Eigen::VectorXd {
    Eigen::VectorXd c = a + b;
    c[6] = tools::limit_rad(c[6]);
    return c;
  };

  ekf_ = tools::ExtendedKalmanFilter(x0, P0, x_add);
}

// ---- 预测 ①：按绝对时间戳推进 -----------------------------------------------
// 外部（tracker/aimer）只知道"现在几点"，不知道"过了多久"，
// 所以由 Target 自己记住上次的时间 t_，算出 dt 再推进，最后更新 t_。
// 这个设计的好处：无论调用方隔了多久调用一次，dt 都是准的。
void Target::predict(std::chrono::steady_clock::time_point t)
{
  auto dt = tools::delta_time(t, t_);  // t - t_，单位秒
  predict(dt);
  t_ = t;
}

// ---- 预测 ②：按时间增量推进（真正干活的那个）--------------------------------
// 两个矩阵决定这一步：
//   F（状态转移）：描述"没有新观测时，状态应该怎么自然演化"
//   Q（过程噪声）：描述"这个演化有多不可信"，Q 越大越不信任模型、越跟观测走
void Target::predict(double dt)
{
  // 状态转移矩阵：纯匀速模型（CV 模型）
  //   位置 = 位置 + 速度·dt（x[0] += x[1]·dt，其余同理）
  //   yaw  = yaw + 角速度·dt（x[6] += x[7]·dt）
  //   r / l / h 视为常数（不随时间变化，所以对角线是 1）
  // 这就是为什么它无法预测"目标突然加速"——只能靠 Q 放宽、靠观测拉回来
  // clang-format off
  Eigen::MatrixXd F{
    {1, dt,  0,  0,  0,  0,  0,  0,  0,  0,  0},
    {0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0},
    {0,  0,  1, dt,  0,  0,  0,  0,  0,  0,  0},
    {0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0},
    {0,  0,  0,  0,  1, dt,  0,  0,  0,  0,  0},
    {0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0},
    {0,  0,  0,  0,  0,  0,  1, dt,  0,  0,  0},
    {0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0},
    {0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0},
    {0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0},
    {0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1}
  };
  // clang-format on

  // 过程噪声：Piecewise White Noise Model（分段白噪声模型）
  // 假设加速度在 dt 内是常数但每步随机变化，积分后得到位置/速度的噪声协方差。
  // 系数 a = dt⁴/4, b = dt³/2, c = dt² 就是这个积分的结果。
  // 具体推导见注释里的链接（Kalman-and-Bayesian-Filters-in-Python 第 7 章）。
  //
  // v1 = 加速度方差，v2 = 角加速度方差：
  //   v1 大 → 认为目标会突然变向（机动性强），滤波器更跟手但更抖
  //   v1 小 → 认为目标匀速运动，轨迹更平滑但转弯时会滞后
  // 普通机器人 v1=100 说明"完全不指望它是匀速的"，
  // 前哨站 v1=10、v2=0.1 是因为它固定在原地只做慢速旋转，模型可信度高得多。
  double v1, v2;
  if (name == ArmorName::outpost) {
    v1 = 10;   // 前哨站加速度方差
    v2 = 0.1;  // 前哨站角加速度方差
  } else {
    v1 = 100;  // 加速度方差
    v2 = 400;  // 角加速度方差
  }
  auto a = dt * dt * dt * dt / 4;
  auto b = dt * dt * dt / 2;
  auto c = dt * dt;
  // 预测过程噪声偏差的方差
  // 只有前三组（x/vx、y/vy、z/vz）和 yaw/w 有噪声项，r/l/h 三行全 0，
  // 对应"半径和高度常数不变"的假设
  // clang-format off
  Eigen::MatrixXd Q{
    {a * v1, b * v1,      0,      0,      0,      0,      0,      0, 0, 0, 0},
    {b * v1, c * v1,      0,      0,      0,      0,      0,      0, 0, 0, 0},
    {     0,      0, a * v1, b * v1,      0,      0,      0,      0, 0, 0, 0},
    {     0,      0, b * v1, c * v1,      0,      0,      0,      0, 0, 0, 0},
    {     0,      0,      0,      0, a * v1, b * v1,      0,      0, 0, 0, 0},
    {     0,      0,      0,      0, b * v1, c * v1,      0,      0, 0, 0, 0},
    {     0,      0,      0,      0,      0,      0, a * v2, b * v2, 0, 0, 0},
    {     0,      0,      0,      0,      0,      0, b * v2, c * v2, 0, 0, 0},
    {     0,      0,      0,      0,      0,      0,      0,      0, 0, 0, 0},
    {     0,      0,      0,      0,      0,      0,      0,      0, 0, 0, 0},
    {     0,      0,      0,      0,      0,      0,      0,      0, 0, 0, 0}
  };
  // clang-format on

  // 非线性状态转移函数 f(x)：先做线性推进，再把角度折回 [-π, π]
  auto f = [&](const Eigen::VectorXd & x) -> Eigen::VectorXd {
    Eigen::VectorXd x_prior = F * x;
    x_prior[6] = tools::limit_rad(x_prior[6]);
    return x_prior;
  };

  // 前哨站转速特判：前哨站结构上转速不可能超过 2 rad/s，
  // 但如果某一帧检测跳变（比如把上层装甲板看成下层），EKF 会把 w 拉到一个
  // 离谱的值，之后就再也拉不回来。这里直接把它夹到 2.51，等于人为"掰回来"。
  // 这是典型的"用一个硬约束兜住滤波器发散"的工程手段。
  if (this->convergened() && this->name == ArmorName::outpost && std::abs(this->ekf_.x[7]) > 2)
    this->ekf_.x[7] = this->ekf_.x[7] > 0 ? 2.51 : -2.51;

  ekf_.predict(F, Q, f);
}

// ---- 更新：先解决"这块装甲板是第几号"，再做 EKF 更新 -----------------------
// 这是整个追踪里最容易出错的一步。原因：
//   观测只告诉我"图像里有一块装甲板在某个位置"，但没告诉我它是车头、车尾还是侧面。
//   而观测方程 h(x) 必须知道"第几号"才能算出预测位置。
// 所以流程是：先用预测的 4 块装甲板位置去和观测比对，猜出 id，再用这个 id 更新。
void Target::update(const Armor & armor)
{
  // 装甲板匹配
  int id;
  auto min_angle_error = 1e10;
  // 用当前状态推算 4 块装甲板此刻应该在哪（x, y, z, angle）
  const std::vector<Eigen::Vector4d> & xyza_list = armor_xyza_list();

  std::vector<std::pair<Eigen::Vector4d, int>> xyza_i_list;
  for (int i = 0; i < armor_num_; i++) {
    xyza_i_list.push_back({xyza_list[i], i});
  }

  // 第一层筛选：按距离排序，只在前 3 个里找
  // 理由：远处的装甲板投影误差大、匹配最不可靠；而且真正可能出现的那块
  // 一定是离相机最近的几块之一。这个剪枝把匹配范围从 4 缩到 3，省一点算力。
  std::sort(
    xyza_i_list.begin(), xyza_i_list.end(),
    [](const std::pair<Eigen::Vector4d, int> & a, const std::pair<Eigen::Vector4d, int> & b) {
      Eigen::Vector3d ypd1 = tools::xyz2ypd(a.first.head(3));
      Eigen::Vector3d ypd2 = tools::xyz2ypd(b.first.head(3));
      return ypd1[2] < ypd2[2];  // ypd[2] 是距离
    });

  // 第二层筛选：在候选里选"角度最接近"的那块
  // 误差 = |观测的装甲板朝向 - 预测的装甲板朝向| + |观测的方位角 - 预测的方位角|
  // 两项都用 limit_rad 处理过，避免 ±π 边界上的假误差
  for (int i = 0; i < 3; i++) {
    const auto & xyza = xyza_i_list[i].first;
    Eigen::Vector3d ypd = tools::xyz2ypd(xyza.head(3));
    auto angle_error = std::abs(tools::limit_rad(armor.ypr_in_world[0] - xyza[3])) +
                       std::abs(tools::limit_rad(armor.ypd_in_world[0] - ypd[0]));

    if (std::abs(angle_error) < std::abs(min_angle_error)) {
      id = xyza_i_list[i].second;
      min_angle_error = angle_error;
    }
  }

  // id != 0 说明这一帧匹配到的不再是"0 号装甲板"，也就是说观测在几块装甲板
  // 之间跳了。这是个重要信号：后面的 Aimer 会用它决定要不要相信当前瞄准点
  // （jumped 为 true 时，4 块板的位置才都被"确认"过，才敢算其它板的位置）。
  if (id != 0) jumped = true;

  if (id != last_id) {
    is_switch_ = true;   // 这一帧换了装甲板
  } else {
    is_switch_ = false;
  }

  if (is_switch_) switch_count_++;  // 统计切换次数，用于调试分析

  last_id = id;
  update_count_++;  // 用于判断是否已收敛（见 convergened）

  update_ypda(armor, id);
}

// ---- 真正的 EKF 更新：观测方程 h(x) 与自适应噪声 R --------------------------
// 观测 z 是 4 维：[方位角 yaw, 俯仰角 pitch, 距离, 装甲板朝向角]
// 用球坐标而不是笛卡尔坐标做观测，是因为"距离"和"角度"的测量噪声
// 更接近相互独立、各向同性——笛卡尔坐标下远处目标的 x/y 噪声是强耦合的。
void Target::update_ypda(const Armor & armor, int id)
{
  // 观测雅可比 H：h(x) 对 x 的偏导，由 h_jacobian() 算（见文件末尾）
  Eigen::MatrixXd H = h_jacobian(ekf_.x, id);

  // ---- 自适应观测噪声 R（这份代码里比较有意思的一处）----
  // 前两项（方位角、俯仰角）用固定方差 4e-3；
  // 第三项（距离）随"装甲板相对车心的偏角"增大而增大：
  //   delta_angle 越大 → 我们看到的装甲板越"侧" → 单目测距越不准
  //   用 log(|Δangle|+1)+1 做软约束，避免侧面时距离被过度信任
  // 第四项（装甲板朝向角）随距离增大而增大：
  //   越远，yaw 越不可观，log(distance+1)/200 让远处观测权重自动降低
  auto center_yaw = std::atan2(armor.xyz_in_world[1], armor.xyz_in_world[0]);
  auto delta_angle = tools::limit_rad(armor.ypr_in_world[0] - center_yaw);
  Eigen::VectorXd R_dig{
    {4e-3, 4e-3, log(std::abs(delta_angle) + 1) + 1,
     log(std::abs(armor.ypd_in_world[2]) + 1) / 200 + 9e-2}};

  //测量过程噪声偏差的方差
  Eigen::MatrixXd R = R_dig.asDiagonal();

  // 定义非线性转换函数 h: x -> z
  // 把"整车状态"先算成第 id 块装甲板的位置（h_armor_xyz），
  // 再转成球坐标（xyz2ypd），最后拼上这块装甲板的朝向角
  auto h = [&](const Eigen::VectorXd & x) -> Eigen::Vector4d {
    Eigen::VectorXd xyz = h_armor_xyz(x, id);
    Eigen::VectorXd ypd = tools::xyz2ypd(xyz);
    auto angle = tools::limit_rad(x[6] + id * 2 * CV_PI / armor_num_);
    return {ypd[0], ypd[1], ypd[2], angle};
  };

  // 观测残差要自定义减法：角度是周期量，直接相减会在 ±π 处出现 2π 跳变
  auto z_subtract = [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) -> Eigen::VectorXd {
    Eigen::VectorXd c = a - b;
    c[0] = tools::limit_rad(c[0]);
    c[1] = tools::limit_rad(c[1]);
    c[3] = tools::limit_rad(c[3]);
    return c;
  };

  // 实际观测：detector 给的球坐标 + 装甲板 yaw
  const Eigen::VectorXd & ypd = armor.ypd_in_world;
  const Eigen::VectorXd & ypr = armor.ypr_in_world;
  Eigen::VectorXd z{{ypd[0], ypd[1], ypd[2], ypr[0]}};  //获得观测量

  ekf_.update(z, H, R, h, z_subtract);
}

// 返回当前状态向量（Aimer 选瞄准点、Planner 做 MPC 都靠它）
Eigen::VectorXd Target::ekf_x() const { return ekf_.x; }

// 返回滤波器本体（供 tracker 做 NIS 卡方检验、读取收敛指标）
const tools::ExtendedKalmanFilter & Target::ekf() const { return ekf_; }

// ---- 由状态推算 4 块装甲板的位置 --------------------------------------------
// 这是"整车模型"的价值所在：即使某一帧只看见 1 块板，也能算出另外 3 块在哪。
// 返回的每个元素是 (x, y, z, angle)，angle 是该装甲板的朝向。
std::vector<Eigen::Vector4d> Target::armor_xyza_list() const
{
  std::vector<Eigen::Vector4d> _armor_xyza_list;

  for (int i = 0; i < armor_num_; i++) {
    // 第 i 块装甲板的角度 = 基准角度 + i × (2π / 装甲板数)，均匀分布
    auto angle = tools::limit_rad(ekf_.x[6] + i * 2 * CV_PI / armor_num_);
    Eigen::Vector3d xyz = h_armor_xyz(ekf_.x, i);
    _armor_xyza_list.push_back({xyz[0], xyz[1], xyz[2], angle});
  }
  return _armor_xyza_list;
}

// ---- 发散检测：用物理合理性做兜底 -------------------------------------------
// 唯一的判据是"半径是否合理"：整车中心到装甲板 5cm~50cm。
// 超出这个范围说明滤波器已经跑飞（比如把远处另一台车的数据混进来了），
// 此时任何预测都不可信，tracker 会直接丢弃这个目标重新开始。
// 这比检查"残差是否过大"更鲁棒，因为它不依赖噪声模型是否准确。
bool Target::diverged() const
{
  auto r_ok = ekf_.x[8] > 0.05 && ekf_.x[8] < 0.5;
  auto l_ok = ekf_.x[8] + ekf_.x[9] > 0.05 && ekf_.x[8] + ekf_.x[9] < 0.5;

  if (r_ok && l_ok) return false;

  tools::logger()->debug("[Target] r={:.3f}, l={:.3f}", ekf_.x[8], ekf_.x[9]);
  return true;
}

// ---- 收敛判断：更新够多次、且没发散，才认为"这个目标的估计可信" -------------
// 为什么需要它：刚初始化的头几帧，速度/角速度都还是 0，预测完全不准。
// 这时候如果就去瞄准，会打飞。所以要先"养"几帧。
// 前哨站要求 10 帧（因为它的模型更特殊、初始误差更大），普通目标 3 帧。
bool Target::convergened()
{
  if (this->name != ArmorName::outpost && update_count_ > 3 && !this->diverged()) {
    is_converged_ = true;
  }

  //前哨站特殊判断
  if (this->name == ArmorName::outpost && update_count_ > 10 && !this->diverged()) {
    is_converged_ = true;
  }

  return is_converged_;
}

// ---- 观测方程的核心：状态 → 第 id 块装甲板的三维坐标 ------------------------
// 公式：armor = center - r_id · (cos a_id, sin a_id)，高度用 cz 或 cz + h
// 注意这里用的是"减"，和构造函数里反推中心时的"加"正好相反。
// 如果哪天你发现追踪一直在漂，第一件事就是检查这两处的符号是否仍然一致。
//
// use_l_h 的意义：4 块装甲板的情况下，1 号和 3 号（相隔 90° 的那一对）
// 使用 r + l 和 cz + h，用来描述非对称车型。
// 装甲板中心的坐标（考虑长短轴）
Eigen::Vector3d Target::h_armor_xyz(const Eigen::VectorXd & x, int id) const
{
  auto angle = tools::limit_rad(x[6] + id * 2 * CV_PI / armor_num_);
  auto use_l_h = (armor_num_ == 4) && (id == 1 || id == 3);

  auto r = (use_l_h) ? x[8] + x[9] : x[8];
  auto armor_x = x[0] - r * std::cos(angle);
  auto armor_y = x[2] - r * std::sin(angle);
  auto armor_z = (use_l_h) ? x[4] + x[10] : x[4];

  return {armor_x, armor_y, armor_z};
}

// ---- 观测雅可比 H：链式法则两次 ---------------------------------------------
// h(x) 是复合函数：状态 → 装甲板 xyz → 球坐标 ypd → 拼上角度
// 所以 H = H_ypda · H_xyza，两段分别求导再相乘。
// 手写雅可比而不是用自动微分，是为了在嵌入式平台上省开销（这一步每帧都跑）。
Eigen::MatrixXd Target::h_jacobian(const Eigen::VectorXd & x, int id) const
{
  auto angle = tools::limit_rad(x[6] + id * 2 * CV_PI / armor_num_);
  auto use_l_h = (armor_num_ == 4) && (id == 1 || id == 3);

  auto r = (use_l_h) ? x[8] + x[9] : x[8];
  // 对角度求偏导：armor_x = cx - r·cos(a) → ∂/∂a = +r·sin(a)
  auto dx_da = r * std::sin(angle);
  auto dy_da = -r * std::cos(angle);

  // 对半径求偏导：∂/∂r = -cos(a)
  auto dx_dr = -std::cos(angle);
  auto dy_dr = -std::sin(angle);
  // 对 l 求偏导：只有 1/3 号板受影响
  auto dx_dl = (use_l_h) ? -std::cos(angle) : 0.0;
  auto dy_dl = (use_l_h) ? -std::sin(angle) : 0.0;

  // 对 h 求偏导：只有 1/3 号板的高度受影响
  auto dz_dh = (use_l_h) ? 1.0 : 0.0;

  // H_xyza：4×11，把 11 维状态映射到装甲板 (x, y, z, angle)
  // clang-format off
  Eigen::MatrixXd H_armor_xyza{
    {1, 0, 0, 0, 0, 0, dx_da, 0, dx_dr, dx_dl,     0},
    {0, 0, 1, 0, 0, 0, dy_da, 0, dy_dr, dy_dl,     0},
    {0, 0, 0, 0, 1, 0,     0, 0,     0,     0, dz_dh},
    {0, 0, 0, 0, 0, 0,     1, 0,     0,     0,     0}
  };
  // clang-format on

  // H_ypda：4×4，把 (x, y, z, angle) 映射到 (yaw, pitch, distance, angle)
  // xyz → 球坐标的雅可比由 tools::xyz2ypd_jacobian 提供；
  // 最后一行的 angle 是直接透传（观测的第 4 维就是角度，不需要变换）
  Eigen::VectorXd armor_xyz = h_armor_xyz(x, id);
  Eigen::MatrixXd H_armor_ypd = tools::xyz2ypd_jacobian(armor_xyz);
  // clang-format off
  Eigen::MatrixXd H_armor_ypda{
    {H_armor_ypd(0, 0), H_armor_ypd(0, 1), H_armor_ypd(0, 2), 0},
    {H_armor_ypd(1, 0), H_armor_ypd(1, 1), H_armor_ypd(1, 2), 0},
    {H_armor_ypd(2, 0), H_armor_ypd(2, 1), H_armor_ypd(2, 2), 0},
    {                0,                 0,                 0, 1}
  };
  // clang-format on

  return H_armor_ypda * H_armor_xyza;  // 最终得到 4×11 的观测雅可比
}

// 调试用：是否已完成初始化（实际未被调用）
bool Target::checkinit() { return isinit; }

}  // namespace auto_aim
