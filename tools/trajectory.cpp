// =============================================================================
// trajectory.cpp —— 弹道解算（给定弹速、水平距离、高度差，求 pitch 和飞行时间）
// =============================================================================
// 【这个类解决什么问题】
//   云台要打中 d 米外、高 h 米的目标，pitch 该抬多少度？子弹飞多久？
//   如果忽略空气阻力，这是一个纯抛物运动问题，有解析解（不用迭代）。
//
// 【数学推导】（看懂这段，这个文件就完全明白了）
//   设出膛速度 v0，pitch 角 θ，飞行时间 t：
//       水平： d = v0·cosθ · t            =>  t = d / (v0·cosθ)
//       竖直： h = v0·sinθ · t - g·t² / 2
//   把 t 代进竖直方程，两边同除 d，并用 1/cos²θ = 1 + tan²θ 整理，得：
//       (g·d² / (2·v0²)) · tan²θ  -  d·tanθ  +  (g·d²/(2·v0²) + h)  =  0
//   这正是关于 tanθ 的一元二次方程 ax² + bx + c = 0，其中：
//       a = g·d² / (2·v0²)
//       b = -d
//       c = a + h
//
//   判别式 Δ = b² - 4ac：
//     Δ < 0  → 打不到（初速不够，或目标太高/太远），unsolvable = true
//     Δ ≥ 0  → 两个解，对应"高抛"和"平射"两条弹道
//
// 【为什么选飞行时间短的那个】
//   两个解都能命中静止目标，但平射弹道（t 小）更好：
//     1. 飞行时间短 → 目标在这段时间内移动少 → 预测误差小
//     2. 弹道平 → 同样的角度误差造成的落点偏差更小（灵敏度低）
//   所以代码里用 t_1 < t_2 做选择，而不是按角度大小选。
// =============================================================================

#include "trajectory.hpp"

#include <cmath>

namespace tools
{
// 上海地区的重力加速度实测值（标准值 9.80665，本地略小）。
// 这个常数会影响远距离命中，不同赛区最好按当地纬度改。
constexpr double g = 9.7833;

// 构造函数：传入即解算，结果直接放在成员变量 pitch / fly_time / unsolvable 里
//   参数 v0 ：子弹出膛速度（m/s），来自下位机实时上报
//   参数 d  ：目标在水平面上的投影距离（m）
//   参数 h  ：目标相对枪口的高度差（m），向上为正
Trajectory::Trajectory(const double v0, const double d, const double h)
{
  // --- 构造一元二次方程 a·tan²θ + b·tanθ + c = 0 的三个系数 ---
  auto a = g * d * d / (2 * v0 * v0);
  auto b = -d;
  auto c = a + h;
  auto delta = b * b - 4 * a * c;  // 判别式

  // --- 判别式小于 0：物理上打不到 ---
  // 触发条件通常是：弹速掉太多（摩擦轮掉速）、目标太远、仰角不够。
  // 上层（Aimer）会检查 unsolvable 并放弃这一帧，避免输出错误的 pitch。
  if (delta < 0) {
    unsolvable = true;
    return;
  }

  unsolvable = false;

  // --- 两个解：高抛弹道 和 平射弹道 ---
  auto tan_pitch_1 = (-b + std::sqrt(delta)) / (2 * a);
  auto tan_pitch_2 = (-b - std::sqrt(delta)) / (2 * a);
  auto pitch_1 = std::atan(tan_pitch_1);  // atan 把 tanθ 还原成角度（弧度）
  auto pitch_2 = std::atan(tan_pitch_2);

  // --- 各自的飞行时间：t = d / (v0·cosθ) ---
  auto t_1 = d / (v0 * std::cos(pitch_1));
  auto t_2 = d / (v0 * std::cos(pitch_2));

  // --- 取飞行时间更短的那条（通常是平射） ---
  // 顺便把 pitch 和 fly_time 一起定下来，两者必须来自同一个解，不能混搭
  pitch = (t_1 < t_2) ? pitch_1 : pitch_2;
  fly_time = (t_1 < t_2) ? t_1 : t_2;
}

}  // namespace tools
