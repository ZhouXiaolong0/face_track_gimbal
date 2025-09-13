#ifndef FACE_TRACK_GIMBAL_FACE_TRACK_CTRL_FPS_COUNTER_H_
#define FACE_TRACK_GIMBAL_FACE_TRACK_CTRL_FPS_COUNTER_H_

#include <chrono>
#include <deque>

/**
 * @brief 滑动窗口平均 FPS 计算器
 *
 * 用于平滑地估计帧率，避免逐帧计算时数值波动过大。
 *
 * 使用方式：
 * @code
 * FPSCounter fps(30);   // 使用窗口大小为 30
 * while (true) {
 *     fps.tick();       // 每帧调用一次
 *     double curFPS = fps.getFPS();
 * }
 * @endcode
 */
class FPSCounter {
public:
  /**
   * @brief 构造函数
   * @param window_size 滑动窗口大小（默认 30 帧）
   */
  explicit FPSCounter(size_t window_size = 30);

  /**
   * @brief 在处理完一帧后调用，用于记录时间戳
   */
  void tick();

  /**
   * @brief 获取平滑后的 FPS
   * @return 当前估算的 FPS 值
   */
  double getFPS() const;

private:
  size_t window_size_; ///< 滑动窗口大小
  std::deque<std::chrono::steady_clock::time_point> timestamps_; ///< 时间戳队列
};

#endif // FACE_TRACK_GIMBAL_FACE_TRACK_CTRL_FPS_COUNTER_H_
