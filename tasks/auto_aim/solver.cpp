// =============================================================================
// solver.cpp —— 位姿解算：像素坐标 → 世界坐标，以及单目 yaw 的修正
// =============================================================================
// 【职责】
//   1. 维护"云台系 → 世界系"的旋转（靠 IMU 四元数驱动）
//   2. solve()：把一张图里检测到的装甲板四角点，用 PnP 解出它在世界系的三维坐标
//   3. optimize_yaw()：修掉单目相机对"装甲板朝向"的观测歧义（这是本文件的核心）
//
// 【为什么需要 optimize_yaw —— 理解这一点的前提】
//   单目相机看一个平面（装甲板），只靠 4 个角点求位姿，存在"翻转歧义"：
//   同一个投影可能对应两种朝向（正面 / 背面），PnP 无法分辨。
//   而且装甲板的 yaw 对重投影误差的影响很小（平面绕自身法线转，投影变化微弱），
//   所以 PnP 给出的 yaw 抖得很厉害，直接拿去做预瞄会飞。
//
//   本项目采用的办法：固定其它自由度，只让 yaw 在合理范围内变化，
//   看哪个 yaw 能让"反投影回去的 4 个点"和"检测到的 4 个点"最贴合。
//   这不是最优估计，但是个简单有效的工程解。
//
// 【三个坐标系】（后面所有变换都在这三个里打转）
//   camera ：相机系，z 轴沿光轴向前
//   gimbal ：云台系，随云台转动
//   world  ：世界系，与 IMU 绝对姿态一致（近似水平面为 xy 平面）
// =============================================================================

#include "solver.hpp"

#include <yaml-cpp/yaml.h>

#include <vector>

#include "tools/logger.hpp"
#include "tools/math_tools.hpp"

namespace auto_aim
{
// ---- 装甲板的物理尺寸（单位：米）---------------------------------------------
// LIGHTBAR_LENGTH：灯条长度，即装甲板的高度方向尺寸
// BIG/SMALL_ARMOR_WIDTH：两块灯条之间的间距（大装甲板 / 小装甲板）
// 这三个数必须和实际机器人一致，否则解出来的距离会系统性偏大或偏小
constexpr double LIGHTBAR_LENGTH = 56e-3;     // m
constexpr double BIG_ARMOR_WIDTH = 230e-3;    // m
constexpr double SMALL_ARMOR_WIDTH = 135e-3;  // m

// ---- 装甲板模型点（装甲板自身坐标系下的 4 个角点）----------------------------
// 注意这个坐标系的约定：x 轴是装甲板的法向（朝外），所以所有点 x = 0；
// y 轴沿灯条间距方向，z 轴沿灯条长度方向。
// 顺序：左上 → 右上 → 右下 → 左下，必须和 detector 输出的 keypoints 顺序一致，
// 否则 PnP 会解出一个镜像的姿态。
const std::vector<cv::Point3f> BIG_ARMOR_POINTS{
  {0, BIG_ARMOR_WIDTH / 2, LIGHTBAR_LENGTH / 2},
  {0, -BIG_ARMOR_WIDTH / 2, LIGHTBAR_LENGTH / 2},
  {0, -BIG_ARMOR_WIDTH / 2, -LIGHTBAR_LENGTH / 2},
  {0, BIG_ARMOR_WIDTH / 2, -LIGHTBAR_LENGTH / 2}};
const std::vector<cv::Point3f> SMALL_ARMOR_POINTS{
  {0, SMALL_ARMOR_WIDTH / 2, LIGHTBAR_LENGTH / 2},
  {0, -SMALL_ARMOR_WIDTH / 2, LIGHTBAR_LENGTH / 2},
  {0, -SMALL_ARMOR_WIDTH / 2, -LIGHTBAR_LENGTH / 2},
  {0, SMALL_ARMOR_WIDTH / 2, -LIGHTBAR_LENGTH / 2}};

// ---- 构造函数：一次性读入所有标定参数 ---------------------------------------
// 这些参数都由 calibration/ 目录下的程序产出，写进 yaml，这里只负责读
Solver::Solver(const std::string & config_path) : R_gimbal2world_(Eigen::Matrix3d::Identity())
{
  auto yaml = YAML::LoadFile(config_path);

  // ① 云台系 → IMU 机体系的固定旋转：IMU 的安装朝向和云台不一定同轴，
  //    这个矩阵把"云台坐标系"换到"IMU 机体坐标系"，后面才能用 IMU 四元数
  auto R_gimbal2imubody_data = yaml["R_gimbal2imubody"].as<std::vector<double>>();
  // ② 相机 → 云台的外参（旋转 + 平移）：相机装歪了、装偏了都体现在这里，
  //    也就是通常说的"手眼标定"结果
  auto R_camera2gimbal_data = yaml["R_camera2gimbal"].as<std::vector<double>>();
  auto t_camera2gimbal_data = yaml["t_camera2gimbal"].as<std::vector<double>>();
  // 注意 RowMajor：yaml 里是按行写的 9 个数，Eigen 默认列优先，必须显式声明
  R_gimbal2imubody_ = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>(R_gimbal2imubody_data.data());
  R_camera2gimbal_ = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>(R_camera2gimbal_data.data());
  t_camera2gimbal_ = Eigen::Matrix<double, 3, 1>(t_camera2gimbal_data.data());

  // ③ 相机内参矩阵（fx, fy, cx, cy 组成的 3×3）和畸变系数（k1,k2,p1,p2,k3）
  auto camera_matrix_data = yaml["camera_matrix"].as<std::vector<double>>();
  auto distort_coeffs_data = yaml["distort_coeffs"].as<std::vector<double>>();
  Eigen::Matrix<double, 3, 3, Eigen::RowMajor> camera_matrix(camera_matrix_data.data());
  Eigen::Matrix<double, 1, 5> distort_coeffs(distort_coeffs_data.data());
  // Eigen 类型转成 OpenCV 的 Mat，因为 solvePnP/projectPoints 只认 cv::Mat
  cv::eigen2cv(camera_matrix, camera_matrix_);
  cv::eigen2cv(distort_coeffs, distort_coeffs_);
}

// 返回当前"云台系 → 世界系"的旋转矩阵（供其它模块换算用）
Eigen::Matrix3d Solver::R_gimbal2world() const { return R_gimbal2world_; }

// ---- 用 IMU 四元数更新"云台系 → 世界系" -------------------------------------
// 公式是一个相似变换（共轭）：R_g2w = R_g2ib^T · R_ib2abs · R_g2ib
// 含义：先把云台系转到 IMU 机体系，再用 IMU 的绝对姿态转到世界系，
//       最后转回云台系的表达方式。
// 可以理解为："IMU 测的是它自己怎么转的，我要知道云台怎么转的，中间差一个安装角"
void Solver::set_R_gimbal2world(const Eigen::Quaterniond & q)
{
  Eigen::Matrix3d R_imubody2imuabs = q.toRotationMatrix();
  R_gimbal2world_ = R_gimbal2imubody_.transpose() * R_imubody2imuabs * R_gimbal2imubody_;
}

// ---- 核心函数 ①：单块装甲板的位姿解算（像素 → 世界）--------------------------
// 输入：armor.points（4 个像素角点）、armor.type（决定用哪套模型点）
// 输出：armor.xyz_in_gimbal / xyz_in_world（位置）、ypr_in_gimbal / ypr_in_world（姿态）
// 副作用：会调用 optimize_yaw() 覆盖 armor.ypr_in_world[0]
void Solver::solve(Armor & armor) const
{
  // 按大/小装甲板选模型点。选错会直接导致距离解算错一个量级，所以
  // detector 那边的 check_type() 也会做一致性校验。
  const auto & object_points =
    (armor.type == ArmorType::big) ? BIG_ARMOR_POINTS : SMALL_ARMOR_POINTS;

  // IPPE 是专门针对"平面目标 + 4 个共面点"的 PnP 算法，比通用 SOLVEPNP_ITERATIVE
  // 快得多，也没有初值依赖——非常适合每帧对每块装甲板都算一遍。
  cv::Vec3d rvec, tvec;
  cv::solvePnP(
    object_points, armor.points, camera_matrix_, distort_coeffs_, rvec, tvec, false,
    cv::SOLVEPNP_IPPE);

  // 位置：camera 系 → gimbal 系 → world 系（连续两次刚体变换）
  Eigen::Vector3d xyz_in_camera;
  cv::cv2eigen(tvec, xyz_in_camera);
  armor.xyz_in_gimbal = R_camera2gimbal_ * xyz_in_camera + t_camera2gimbal_;
  armor.xyz_in_world = R_gimbal2world_ * armor.xyz_in_gimbal;

  // 姿态：rvec（旋转向量）→ 旋转矩阵 → 逐级转到 world 系
  cv::Mat rmat;
  cv::Rodrigues(rvec, rmat);  // 罗德里格斯公式：旋转向量 ⇄ 旋转矩阵
  Eigen::Matrix3d R_armor2camera;
  cv::cv2eigen(rmat, R_armor2camera);
  Eigen::Matrix3d R_armor2gimbal = R_camera2gimbal_ * R_armor2camera;
  Eigen::Matrix3d R_armor2world = R_gimbal2world_ * R_armor2gimbal;
  // 欧拉角顺序 2,1,0 即 yaw-pitch-roll；本项目只关心 yaw 和 pitch
  armor.ypr_in_gimbal = tools::eulers(R_armor2gimbal, 2, 1, 0);
  armor.ypr_in_world = tools::eulers(R_armor2world, 2, 1, 0);

  // 同时缓存一份球坐标（yaw, pitch, distance），EKF 的观测方程用的就是这个表示，
  // 因为球坐标下"距离"和"角度"的噪声特性更接近各向同性
  armor.ypd_in_world = tools::xyz2ypd(armor.xyz_in_world);

  // 平衡机器人（步兵/英雄的平衡底盘形态）的装甲板是倾斜安装的，
  // optimize_yaw 里"装甲板 pitch 固定为 ±15°"的假设不成立，所以跳过优化
  auto is_balance = (armor.type == ArmorType::big) &&
                    (armor.name == ArmorName::three || armor.name == ArmorName::four ||
                     armor.name == ArmorName::five);
  if (is_balance) return;

  optimize_yaw(armor);  // 用重投影搜索修正 yaw（见下方详解）
}

// ---- 核心函数 ②：把"假设的位姿"反投影回图像 ---------------------------------
// 这是 optimize_yaw 的内循环。给定世界坐标 + 一个候选 yaw，
// 构造装甲板姿态 → 转到相机系 → projectPoints → 得到 4 个预测像素点。
// 与真实检测点比较，误差越小说明这个 yaw 越可信。
std::vector<cv::Point2f> Solver::reproject_armor(
  const Eigen::Vector3d & xyz_in_world, double yaw, ArmorType type, ArmorName name) const
{
  auto sin_yaw = std::sin(yaw);
  auto cos_yaw = std::cos(yaw);

  // 装甲板的安装 pitch 是固定的：普通装甲板向上倾 15°，前哨站向下倾 15°。
  // 这是机械结构决定的常数，所以这里不需要估计，直接代入。
  auto pitch = (name == ArmorName::outpost) ? -15.0 * CV_PI / 180.0 : 15.0 * CV_PI / 180.0;
  auto sin_pitch = std::sin(pitch);
  auto cos_pitch = std::cos(pitch);

  // 由 yaw + pitch 构造"装甲板 → 世界"的旋转矩阵（列主序手写，省一次欧拉角计算）
  // clang-format off
  const Eigen::Matrix3d R_armor2world {
    {cos_yaw * cos_pitch, -sin_yaw, cos_yaw * sin_pitch},
    {sin_yaw * cos_pitch,  cos_yaw, sin_yaw * sin_pitch},
    {         -sin_pitch,        0,           cos_pitch}
  };
  // clang-format on

  // world → camera：先转回 gimbal，再转回 camera；平移要用"相机在世界的位姿"反算
  const Eigen::Vector3d & t_armor2world = xyz_in_world;
  Eigen::Matrix3d R_armor2camera =
    R_camera2gimbal_.transpose() * R_gimbal2world_.transpose() * R_armor2world;
  Eigen::Vector3d t_armor2camera =
    R_camera2gimbal_.transpose() * (R_gimbal2world_.transpose() * t_armor2world - t_camera2gimbal_);

  // 旋转矩阵转回旋转向量，交给 OpenCV 做投影
  cv::Vec3d rvec;
  cv::Mat R_armor2camera_cv;
  cv::eigen2cv(R_armor2camera, R_armor2camera_cv);
  cv::Rodrigues(R_armor2camera_cv, rvec);
  cv::Vec3d tvec(t_armor2camera[0], t_armor2camera[1], t_armor2camera[2]);

  // 带畸变模型的投影，和相机的真实成像一致，否则边缘区域的误差会被高估
  std::vector<cv::Point2f> image_points;
  const auto & object_points = (type == ArmorType::big) ? BIG_ARMOR_POINTS : SMALL_ARMOR_POINTS;
  cv::projectPoints(object_points, rvec, tvec, camera_matrix_, distort_coeffs_, image_points);
  return image_points;
}

// ---- 前哨站专用：给定 pitch，算重投影误差 -----------------------------------
// 前哨站有两层装甲板（上 27.5° / 下 -15°），单张图分不清是哪一层，
// 所以用"两种 pitch 各算一次误差"来判别当前看到的是上层还是下层。
// 注意：这个函数按值接收 Armor（会拷贝），而且内部重复了一遍 solve 的逻辑，
//      属于典型的"为调试临时加的代码"，正式流程里没被调用（tracker 里是注释状态）
double Solver::oupost_reprojection_error(Armor armor, const double & pitch)
{
  // solve
  const auto & object_points =
    (armor.type == ArmorType::big) ? BIG_ARMOR_POINTS : SMALL_ARMOR_POINTS;

  cv::Vec3d rvec, tvec;
  cv::solvePnP(
    object_points, armor.points, camera_matrix_, distort_coeffs_, rvec, tvec, false,
    cv::SOLVEPNP_IPPE);

  Eigen::Vector3d xyz_in_camera;
  cv::cv2eigen(tvec, xyz_in_camera);
  armor.xyz_in_gimbal = R_camera2gimbal_ * xyz_in_camera + t_camera2gimbal_;
  armor.xyz_in_world = R_gimbal2world_ * armor.xyz_in_gimbal;

  cv::Mat rmat;
  cv::Rodrigues(rvec, rmat);
  Eigen::Matrix3d R_armor2camera;
  cv::cv2eigen(rmat, R_armor2camera);
  Eigen::Matrix3d R_armor2gimbal = R_camera2gimbal_ * R_armor2camera;
  Eigen::Matrix3d R_armor2world = R_gimbal2world_ * R_armor2gimbal;
  armor.ypr_in_gimbal = tools::eulers(R_armor2gimbal, 2, 1, 0);
  armor.ypr_in_world = tools::eulers(R_armor2world, 2, 1, 0);

  armor.ypd_in_world = tools::xyz2ypd(armor.xyz_in_world);

  auto yaw = armor.ypr_in_world[0];
  auto xyz_in_world = armor.xyz_in_world;

  auto sin_yaw = std::sin(yaw);
  auto cos_yaw = std::cos(yaw);

  auto sin_pitch = std::sin(pitch);
  auto cos_pitch = std::cos(pitch);

  // clang-format off
  const Eigen::Matrix3d _R_armor2world {
    {cos_yaw * cos_pitch, -sin_yaw, cos_yaw * sin_pitch},
    {sin_yaw * cos_pitch,  cos_yaw, sin_yaw * sin_pitch},
    {         -sin_pitch,        0,           cos_pitch}
  };
  // clang-format on

  const Eigen::Vector3d & t_armor2world = xyz_in_world;
  Eigen::Matrix3d _R_armor2camera =
    R_camera2gimbal_.transpose() * R_gimbal2world_.transpose() * _R_armor2world;
  Eigen::Vector3d t_armor2camera =
    R_camera2gimbal_.transpose() * (R_gimbal2world_.transpose() * t_armor2world - t_camera2gimbal_);

  cv::Vec3d _rvec;
  cv::Mat R_armor2camera_cv;
  cv::eigen2cv(_R_armor2camera, R_armor2camera_cv);
  cv::Rodrigues(R_armor2camera_cv, _rvec);
  cv::Vec3d _tvec(t_armor2camera[0], t_armor2camera[1], t_armor2camera[2]);

  std::vector<cv::Point2f> image_points;
  cv::projectPoints(object_points, _rvec, _tvec, camera_matrix_, distort_coeffs_, image_points);

  auto error = 0.0;
  for (int i = 0; i < 4; i++) error += cv::norm(armor.points[i] - image_points[i]);
  return error;
}

// ---- 核心函数 ③：yaw 网格搜索（本文件最重的计算，也是最大的性能热点）-------
// 思路：在"云台当前朝向 ±70°"这个范围内，以 1° 为步长试 140 个 yaw，
//       每个 yaw 都反投影一次，算 4 个点的像素误差之和，取最小的那个。
//
// 为什么以云台朝向为中心：目标基本都在视野里，而视野的 yaw 范围大致就是
//   云台朝向 ±(FOV/2)，用这个先验可以把搜索范围砍掉一大半。
//
// 代价：140 次 projectPoints + 140 次 Rodrigues + 420 次矩阵乘，
//   每块装甲板每帧都要跑一遍。这是后续最值得优化的地方（见架构分析 B1）：
//   改成"粗到细"两步搜索（15° 扫 10 次 + 3° 扫 10 次）精度不变，开销降到 1/5。
void Solver::optimize_yaw(Armor & armor) const
{
  // 取当前云台的 yaw 作为搜索中心
  Eigen::Vector3d gimbal_ypr = tools::eulers(R_gimbal2world_, 2, 1, 0);

  constexpr double SEARCH_RANGE = 140;  // degree，即 ±70°
  auto yaw0 = tools::limit_rad(gimbal_ypr[0] - SEARCH_RANGE / 2 * CV_PI / 180.0);

  auto min_error = 1e10;
  auto best_yaw = armor.ypr_in_world[0];  // 兜底：搜索失败就沿用 PnP 的结果

  for (int i = 0; i < SEARCH_RANGE; i++) {
    double yaw = tools::limit_rad(yaw0 + i * CV_PI / 180.0);
    // 第三个参数 inclined 是"相对搜索中心的偏移"，当前代价函数里没用到，
    // 但 SJTU_cost 用它来加权（见下）
    auto error = armor_reprojection_error(armor, yaw, (i - SEARCH_RANGE / 2) * CV_PI / 180.0);

    if (error < min_error) {
      min_error = error;
      best_yaw = yaw;
    }
  }

  // 保留原始值备查（用于分析 PnP 的 yaw 到底偏多少）
  armor.yaw_raw = armor.ypr_in_world[0];
  armor.ypr_in_world[0] = best_yaw;
}

// ---- 另一种代价函数（当前未启用，被注释在 armor_reprojection_error 里）------
// 思路来自上交的开源代码：不直接比"点与点的像素距离"，而是比"四条边的
// 长度和方向"。好处是对角点检测的局部抖动更鲁棒（角点位置会抖，但边长不会）。
// inclined 是倾斜加权：正对时更信长度，侧对时更信角度。
// 这也说明这份代码参考了 SJTU 的实现（函数名直接叫 SJTU_cost）。
double Solver::SJTU_cost(
  const std::vector<cv::Point2f> & cv_refs, const std::vector<cv::Point2f> & cv_pts,
  const double & inclined) const
{
  std::size_t size = cv_refs.size();
  std::vector<Eigen::Vector2d> refs;
  std::vector<Eigen::Vector2d> pts;
  for (std::size_t i = 0u; i < size; ++i) {
    refs.emplace_back(cv_refs[i].x, cv_refs[i].y);
    pts.emplace_back(cv_pts[i].x, cv_pts[i].y);
  }
  double cost = 0.;
  for (std::size_t i = 0u; i < size; ++i) {
    std::size_t p = (i + 1u) % size;
    // i - p 构成线段。过程：先移动起点，再补长度，再旋转
    Eigen::Vector2d ref_d = refs[p] - refs[i];  // 标准
    Eigen::Vector2d pt_d = pts[p] - pts[i];
    // 长度差代价 + 起点差代价(1 / 2)（0 度左右应该抛弃)
    double pixel_dis =  // dis 是指方差平面内到原点的距离
      (0.5 * ((refs[i] - pts[i]).norm() + (refs[p] - pts[p]).norm()) +
       std::fabs(ref_d.norm() - pt_d.norm())) /
      ref_d.norm();
    double angular_dis = ref_d.norm() * tools::get_abs_angle(ref_d, pt_d) / ref_d.norm();
    // 平方可能是为了配合 sin 和 cos
    // 弧度差代价（0 度左右占比应该大）
    double cost_i =
      tools::square(pixel_dis * std::sin(inclined)) +
      tools::square(angular_dis * std::cos(inclined)) * 2.0;  // DETECTOR_ERROR_PIXEL_BY_SLOPE
    // 重投影像素误差越大，越相信斜率
    cost += std::sqrt(cost_i);
  }
  return cost;
}

// 当前实际使用的代价：4 个角点的像素欧氏距离之和
// 简单粗暴，但对"角点检测本身很准"的神经网络检测结果来说足够了
double Solver::armor_reprojection_error(
  const Armor & armor, double yaw, const double & inclined) const
{
  auto image_points = reproject_armor(armor.xyz_in_world, yaw, armor.type, armor.name);
  auto error = 0.0;
  for (int i = 0; i < 4; i++) error += cv::norm(armor.points[i] - image_points[i]);
  // auto error = SJTU_cost(image_points, armor.points, inclined);  // 备选代价函数

  return error;
}

// ---- 世界坐标 → 像素坐标（画图/可视化用）------------------------------------
// 用于把"预测的装甲板位置"画到实况图上，直观检查跟踪是否跑飞
std::vector<cv::Point2f> Solver::world2pixel(const std::vector<cv::Point3f> & worldPoints)
{
  // world → camera 的完整变换：先反旋转，再反平移
  Eigen::Matrix3d R_world2camera = R_camera2gimbal_.transpose() * R_gimbal2world_.transpose();
  Eigen::Vector3d t_world2camera = -R_camera2gimbal_.transpose() * t_camera2gimbal_;

  cv::Mat rvec;
  cv::Mat tvec;
  cv::eigen2cv(R_world2camera, rvec);
  cv::eigen2cv(t_world2camera, tvec);

  // 只投影相机前方（z > 0）的点，否则会被投影到图像背面变成乱点
  std::vector<cv::Point3f> valid_world_points;
  for (const auto & world_point : worldPoints) {
    Eigen::Vector3d world_point_eigen(world_point.x, world_point.y, world_point.z);
    Eigen::Vector3d camera_point = R_world2camera * world_point_eigen + t_world2camera;

    if (camera_point.z() > 0) {
      valid_world_points.push_back(world_point);
    }
  }
  // 如果没有有效点，返回空vector
  if (valid_world_points.empty()) {
    return std::vector<cv::Point2f>();
  }
  std::vector<cv::Point2f> pixelPoints;
  cv::projectPoints(valid_world_points, rvec, tvec, camera_matrix_, distort_coeffs_, pixelPoints);
  return pixelPoints;
}
}  // namespace auto_aim
