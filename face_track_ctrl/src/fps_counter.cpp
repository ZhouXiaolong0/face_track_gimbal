#include "fps_counter.h"

FPSCounter::FPSCounter(size_t window_size) : window_size_(window_size) {}

void FPSCounter::tick() {
  using namespace std::chrono;
  auto now = steady_clock::now();
  timestamps_.push_back(now);

  // 超过窗口大小时丢弃最早的时间戳
  if (timestamps_.size() > window_size_) {
    timestamps_.pop_front();
  }
}

double FPSCounter::getFPS() const {
  if (timestamps_.size() < 2)
    return 0.0;

  using namespace std::chrono;
  auto duration =
      duration_cast<milliseconds>(timestamps_.back() - timestamps_.front())
          .count();

  if (duration == 0)
    return 0.0;

  // (N-1) 帧耗时 duration 毫秒，FPS = (N-1) / (耗时秒数)
  return (timestamps_.size() - 1) * 1000.0 / duration;
}
