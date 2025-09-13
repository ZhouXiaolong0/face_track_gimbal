#include "gimbal_controller.h"
#include "telemetry/data_logger.h"
#include "telemetry/offset_metric.h"

#include <cmath>

// 一阶低通滤波
static float lowPassFilter(float input, float &prev, float alpha) {
  float output = alpha * input + (1.0f - alpha) * prev;
  prev = output; // 更新状态
  return output;
}

GimbalController::Offset GimbalController::computeOffsets(const cv::Rect &bbox,
                                                          int frame_width,
                                                          int frame_height) {
  int cx = frame_width / 2;
  int cy = frame_height / 2;

  int fx = bbox.x + bbox.width / 2;
  int fy = bbox.y + bbox.height / 2;

  int raw_dx = fx - cx;
  int raw_dy = fy - cy;

  // ---------- 一阶滤波 ----------
  static float prev_dx = 0.0f;
  static float prev_dy = 0.0f;
  const float alpha = 0.3f; // 平滑系数，0.0~1.0，数值越大越灵敏

  float filtered_dx = lowPassFilter(raw_dx, prev_dx, alpha);
  float filtered_dy = lowPassFilter(raw_dy, prev_dy, alpha);

  prev_dx = filtered_dx;
  prev_dy = filtered_dy;

  DataLogger offset_logger("offsets.csv");
  OffsetMetric offset_metric(offset_logger);

  offset_metric.record(raw_dx, raw_dy, static_cast<int>(filtered_dx),
                       static_cast<int>(filtered_dy));

  return {static_cast<int>(filtered_dx), static_cast<int>(filtered_dy)};
}

bool GimbalController::needMove(const GimbalController::Offset &offset,
                                int threshold) {
  return std::abs(offset.dx) >= threshold || std::abs(offset.dy) >= threshold;
}
