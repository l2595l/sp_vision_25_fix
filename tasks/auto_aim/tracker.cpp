// =============================================================================
// tracker.cpp —— 目标管理：状态机 + 目标选择 + 驱动 EKF
// =============================================================================
// 【它和 Target 的分工】
//   Target  只管"一个目标的状态估计"（数学部分，见 target.cpp）
//   Tracker 管"什么时候创建目标、什么时候丢弃目标、多个目标里选哪个"
//           （工程部分，这里全是 if-else 和经验阈值）
//
// 【五个状态】
//   lost      没有目标。每一帧都尝试用检测结果新建一个
//   detecting 刚锁定，正在攒帧数。攒够 min_detect_count 帧才敢进 tracking
//   tracking  正常跟踪。这一帧没检测到就进 temp_lost
//   temp_lost 短暂丢失。容忍 max_temp_lost_count 帧，超了才回到 lost
//   switching 仅哨兵：全向感知看到了更高优先级的目标，正在把云台转过去
//
// 【为什么要有 detecting 这个"攒帧"状态】
//   单帧检测可能有误检。如果第一帧就当成目标开始跟踪，会白算一堆东西，
//   而且 EKF 会被错误初值带偏。要求连续 N 帧都检测到，是廉价的抗误检手段。
// =============================================================================

#include "tracker.hpp"

#include <yaml-cpp/yaml.h>

#include <tuple>

#include "tools/logger.hpp"
#include "tools/math_tools.hpp"

namespace auto_aim
{
// ---- 构造函数：读配置 --------------------------------------------------------
Tracker::Tracker(const std::string & config_path, Solver & solver)
: solver_{solver},
  detect_count_(0),
  temp_lost_count_(0),
  state_{"lost"},
  pre_state_{"lost"},
  last_timestamp_(std::chrono::steady_clock::now()),
  omni_target_priority_{ArmorPriority::fifth}
{
  auto yaml = YAML::LoadFile(config_path);
  // 敌方颜色：决定过滤掉一半的检测结果（红方还是蓝方）
  enemy_color_ = (yaml["enemy_color"].as<std::string>() == "red") ? Color::red : Color::blue;
  // 连续检测到多少帧才认为"锁定成功"
  min_detect_count_ = yaml["min_detect_count"].as<int>();
  // 丢失后容忍多少帧
  max_temp_lost_count_ = yaml["max_temp_lost_count"].as<int>();
  // 前哨站单独放宽（因为它转速慢、容易长时间只看到一小块）
  outpost_max_temp_lost_count_ = yaml["outpost_max_temp_lost_count"].as<int>();
  normal_temp_lost_count_ = max_temp_lost_count_;
}

// 当前状态名，供日志/决策模块查询
std::string Tracker::state() const { return state_; }

// ---- 主函数 ①：普通兵种的目标跟踪 -------------------------------------------
// 输入：本帧所有检测到的装甲板（会被就地修改：过滤、排序、写入世界坐标）
// 输出：0 或 1 个 Target（本项目同一时刻只跟一个目标）
std::list<Target> Tracker::track(
  std::list<Armor> & armors, std::chrono::steady_clock::time_point t, bool use_enemy_color)
{
  // 算与上一帧的间隔。dt 异常大说明中间掉帧/相机断过，
  // 此时历史状态已不可信，直接重置成 lost 重新开始（代价是丢一次目标）
  auto dt = tools::delta_time(t, last_timestamp_);
  last_timestamp_ = t;

  // 时间间隔过长，说明可能发生了相机离线
  if (state_ != "lost" && dt > 0.1) {
    tools::logger()->warn("[Tracker] Large dt: {:.3f}s", dt);
    state_ = "lost";
  }
  // 过滤掉非我方装甲板（只看敌人）
  armors.remove_if([&](const auto_aim::Armor & a) { return a.color != enemy_color_; });

  // 过滤前哨站顶部装甲板（当前关闭）
  // 思路：前哨站上下两层，用两种 pitch 假设各算一次重投影误差，误差小的那个才算对
  // armors.remove_if([this](const auto_aim::Armor & a) {
  //   return a.name == ArmorName::outpost &&
  //          solver_.oupost_reprojection_error(a, 27.5 * CV_PI / 180.0) <
  //            solver_.oupost_reprojection_error(a, -15 * CV_PI / 180.0);
  // });

  // 第一次排序：按"离画面中心的距离"由近到远
  // 注意这里是 list::sort，是稳定排序，所以它的结果会成为下面那次排序的
  // 次级排序键——"同优先级时，取更靠近画面中心的那个"就是这么实现的
  armors.sort([](const Armor & a, const Armor & b) {
    cv::Point2f img_center(1440 / 2, 1080 / 2);  // TODO: 分辨率写死了，换相机会失效
    auto distance_1 = cv::norm(a.center - img_center);
    auto distance_2 = cv::norm(b.center - img_center);
    return distance_1 < distance_2;
  });

  // 第二次排序：按目标优先级（装甲板编号决定，1 最高，如英雄 > 步兵 > 哨兵）
  // 按优先级排序，优先级最高在首位(优先级越高数字越小，1的优先级最高)
  armors.sort(
    [](const auto_aim::Armor & a, const auto_aim::Armor & b) { return a.priority < b.priority; });

  bool found;
  if (state_ == "lost") {
    found = set_target(armors, t);  // 新建目标
  }

  else {
    found = update_target(armors, t);  // 用观测更新已有目标
  }

  state_machine(found);  // 根据 found 推进状态机

  // ---- 三重丢弃条件 ----
  // ① 物理发散：解出来的半径不合理（见 Target::diverged）
  if (state_ != "lost" && target_.diverged()) {
    tools::logger()->debug("[Tracker] Target diverged!");
    state_ = "lost";
    return {};
  }

  // ② 统计意义上的"收敛不佳"：最近 window_size 次更新里，有 40% 以上
  //    没通过 NIS 卡方检验，说明观测和模型长期对不上，滤波器在硬撑
  if (
    std::accumulate(
      target_.ekf().recent_nis_failures.begin(), target_.ekf().recent_nis_failures.end(), 0) >=
    (0.4 * target_.ekf().window_size)) {
    tools::logger()->debug("[Target] Bad Converge Found!");
    state_ = "lost";
    return {};
  }

  // ③ 状态机已经把状态置为 lost
  if (state_ == "lost") return {};

  std::list<Target> targets = {target_};  // 拷贝一份 Target 返回（见架构分析 B4）
  return targets;
}

// ---- 主函数 ②：哨兵专用（带全向感知）----------------------------------------
// 相比上边多了 detection_queue：来自 4 个 USB 相机的"周围有没有更值钱的目标"。
// 战场逻辑：主相机盯着一个人打，但背后出现了英雄，就该立刻转身——
//   这个"转身"就是 switching 状态存在的意义。
// 返回值是 tuple：<需要转身的角度信息, 当前可打的目标列表>
std::tuple<omniperception::DetectionResult, std::list<Target>> Tracker::track(
  const std::vector<omniperception::DetectionResult> & detection_queue, std::list<Armor> & armors,
  std::chrono::steady_clock::time_point t, bool use_enemy_color)
{
  omniperception::DetectionResult switch_target{std::list<Armor>(), t, 0, 0};
  omniperception::DetectionResult temp_target{std::list<Armor>(), t, 0, 0};
  if (!detection_queue.empty()) {
    temp_target = detection_queue.front();  // 全向感知最新一帧的结果
  }

  auto dt = tools::delta_time(t, last_timestamp_);
  last_timestamp_ = t;

  // 时间间隔过长，说明可能发生了相机离线
  if (state_ != "lost" && dt > 0.1) {
    tools::logger()->warn("[Tracker] Large dt: {:.3f}s", dt);
    state_ = "lost";
  }

  // 优先选择靠近图像中心的装甲板
  armors.sort([](const Armor & a, const Armor & b) {
    cv::Point2f img_center(1440 / 2, 1080 / 2);  // TODO
    auto distance_1 = cv::norm(a.center - img_center);
    auto distance_2 = cv::norm(b.center - img_center);
    return distance_1 < distance_2;
  });

  // 按优先级排序，优先级最高在首位(优先级越高数字越小，1的优先级最高)
  armors.sort([](const auto_aim::Armor & a, const auto_aim::Armor & b) {
    return a.priority < b.priority;
  });

  bool found;
  if (state_ == "lost") {
    found = set_target(armors, t);
  }

  // ① 主相机里直接出现了更高优先级的目标：立刻换
  // 此时主相机画面中出现了优先级更高的装甲板，切换目标
  else if (state_ == "tracking" && !armors.empty() && armors.front().priority < target_.priority) {
    found = set_target(armors, t);
    tools::logger()->debug("auto_aim switch target to {}", ARMOR_NAMES[armors.front().name]);
  }

  // ② 只有全向感知看到了更高优先级的目标：进入 switching，把转角信息抛给上层
  //    要求当前目标已收敛（convergened），否则连自己在打谁都没搞清就转身，容易两头空
  // 此时全向感知相机画面中出现了优先级更高的装甲板，切换目标
  else if (
    state_ == "tracking" && !temp_target.armors.empty() &&
    temp_target.armors.front().priority < target_.priority && target_.convergened()) {
    state_ = "switching";
    switch_target = omniperception::DetectionResult{
      temp_target.armors, t, temp_target.delta_yaw, temp_target.delta_pitch};
    omni_target_priority_ = temp_target.armors.front().priority;
    found = false;  // 这一帧不更新 EKF（因为正在转身，主相机的观测不可靠）
    tools::logger()->debug("omniperception find higher priority target");
  }

  // ③ switching 中：主相机终于看到了那个目标，才算切换成功
  else if (state_ == "switching") {
    found = !armors.empty() && armors.front().priority == omni_target_priority_;
  }

  // ④ 刚切换完成的第一帧：直接重建目标，而不是接着旧的 EKF 更新
  else if (state_ == "detecting" && pre_state_ == "switching") {
    found = set_target(armors, t);
  }

  else {
    found = update_target(armors, t);
  }

  pre_state_ = state_;  // 记住切换前的状态，给上面第 ④ 条用
  // 更新状态机
  state_machine(found);

  // 发散检测
  if (state_ != "lost" && target_.diverged()) {
    tools::logger()->debug("[Tracker] Target diverged!");
    state_ = "lost";
    return {switch_target, {}};  // 返回switch_target和空的targets
  }

  if (state_ == "lost") return {switch_target, {}};  // 返回switch_target和空的targets

  std::list<Target> targets = {target_};
  return {switch_target, targets};
}

// ---- 状态机：唯一的输入是"这一帧有没有匹配到目标" ---------------------------
// 所有阈值都来自配置，逻辑本身很朴素，难点在于"每个状态该容忍多少帧"
// 这些数字是拿比赛录像一帧帧调出来的经验值，换兵种要重调。
void Tracker::state_machine(bool found)
{
  if (state_ == "lost") {
    if (!found) return;  // 什么都没有，继续等着

    state_ = "detecting";  // 有东西了，进入考察期
    detect_count_ = 1;
  }

  else if (state_ == "detecting") {
    if (found) {
      detect_count_++;
      if (detect_count_ >= min_detect_count_) state_ = "tracking";  // 考察通过
    } else {
      detect_count_ = 0;
      state_ = "lost";  // 掉一帧就前功尽弃，说明是误检
    }
  }

  else if (state_ == "tracking") {
    if (found) return;  // 正常帧，什么都不做

    temp_lost_count_ = 1;
    state_ = "temp_lost";  // 丢了一帧，开始容忍计数
  }

  else if (state_ == "switching") {
    if (found) {
      state_ = "detecting";  // 转身看到目标了，重新走考察流程
    } else {
      temp_lost_count_++;
      if (temp_lost_count_ > 200) state_ = "lost";  // 转了半天还是没看到，放弃
    }
  }

  else if (state_ == "temp_lost") {
    if (found) {
      state_ = "tracking";  // 又看到了，回到正常
    } else {
      temp_lost_count_++;
      // 前哨站的容忍帧数单独取一组配置（因为它经常长时间只露一小块）
      if (target_.name == ArmorName::outpost)
        //前哨站的temp_lost_count需要设置的大一些
        max_temp_lost_count_ = outpost_max_temp_lost_count_;
      else
        max_temp_lost_count_ = normal_temp_lost_count_;

      if (temp_lost_count_ > max_temp_lost_count_) state_ = "lost";  // 超时，彻底放弃
    }
  }
}

// ---- 新建目标：按兵种给不同的物理参数 ---------------------------------------
// 这里藏着四条重要的"先验知识"，全部是量出来的：
//   radius     ：整车中心到装甲板的距离（米），装错会导致距离解算系统性偏差
//   armor_num  ：车上有几块装甲板（平衡车只有 2 块）
//   P0_dig     ：初始协方差，值越大表示"我越不确定"
//                位置用 1（不确定），速度用 64（非常不确定，因为没有历史），
//                半径用 1e-4（几乎确定，因为是结构尺寸）
bool Tracker::set_target(std::list<Armor> & armors, std::chrono::steady_clock::time_point t)
{
  if (armors.empty()) return false;

  // 取排序后的第一块（优先级最高、且最靠近画面中心）作为初始观测
  auto & armor = armors.front();
  solver_.solve(armor);  // 先补上世界坐标（检测阶段只给了像素坐标）

  // 判断是不是平衡机器人（大装甲板 + 3/4/5 号）
  auto is_balance = (armor.type == ArmorType::big) &&
                    (armor.name == ArmorName::three || armor.name == ArmorName::four ||
                     armor.name == ArmorName::five);

  if (is_balance) {
    // 平衡车：半径 0.2 m，只有 2 块装甲板（前后各一块）
    Eigen::VectorXd P0_dig{{1, 64, 1, 64, 1, 64, 0.4, 100, 1, 1, 1}};
    target_ = Target(armor, t, 0.2, 2, P0_dig);
  }

  else if (armor.name == ArmorName::outpost) {
    // 前哨站：半径 0.2765 m，3 块，半径的初始不确定度极小（1e-4）
    // 因为它结构固定、且转速极慢，几乎不会估错
    Eigen::VectorXd P0_dig{{1, 64, 1, 64, 1, 81, 0.4, 100, 1e-4, 0, 0}};
    target_ = Target(armor, t, 0.2765, 3, P0_dig);
  }

  else if (armor.name == ArmorName::base) {
    // 基地：半径 0.3205 m，3 块
    Eigen::VectorXd P0_dig{{1, 64, 1, 64, 1, 64, 0.4, 100, 1e-4, 0, 0}};
    target_ = Target(armor, t, 0.3205, 3, P0_dig);
  }

  else {
    // 普通机器人：半径 0.2 m，4 块
    Eigen::VectorXd P0_dig{{1, 64, 1, 64, 1, 64, 0.4, 100, 1, 1, 1}};
    target_ = Target(armor, t, 0.2, 4, P0_dig);
  }

  return true;
}

// ---- 更新目标 ---------------------------------------------------------------
// 两个细节值得注意：
//   ① 先 predict(t) 再 update：EKF 的标准顺序，保证状态推进到当前时刻
//   ② 会对"所有同类同型的装甲板"都做一次 update，而不是只取一块
//      因为一帧里可能同时看到同一辆车的两块板（例如正对时能看到左右两块），
//      多块一起更新能让估计更快收敛
bool Tracker::update_target(std::list<Armor> & armors, std::chrono::steady_clock::time_point t)
{
  target_.predict(t);  // 把状态推到当前时刻（用真实 dt）

  // 先数一遍有几块匹配的装甲板；一块都没有就不用 update 了
  int found_count = 0;
  double min_x = 1e10;  // 画面最左侧（当前未被使用）
  for (const auto & armor : armors) {
    if (armor.name != target_.name || armor.type != target_.armor_type) continue;
    found_count++;
    min_x = armor.center.x < min_x ? armor.center.x : min_x;
  }

  if (found_count == 0) return false;

  // 逐个更新。注意 solve() 在这里才被调用——即"只有真正参与跟踪的装甲板
  // 才做 PnP 和 yaw 优化"，这是个有意的性能取舍
  for (auto & armor : armors) {
    if (
      armor.name != target_.name || armor.type != target_.armor_type
      //  || armor.center.x != min_x      // 曾经想只取最左边那块，后来放开了
    )
      continue;

    solver_.solve(armor);

    target_.update(armor);
  }

  return true;
}

}  // namespace auto_aim
