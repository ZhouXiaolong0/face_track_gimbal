#ifndef FACE_TRACK_GIMBAL_FACE_TRACK_CTRL_TELEMETRY_OFFSET_METRIC_H_
#define FACE_TRACK_GIMBAL_FACE_TRACK_CTRL_TELEMETRY_OFFSET_METRIC_H_

#include "data_logger.h"

/**
 * @brief 偏移量度量类，用于记录原始和滤波后的偏移量数据。
 * 
 * 使用 DataLogger 将偏移量数据写入文件，便于分析和调试。
 */
class OffsetMetric {
public:

  /**
   * @brief 构造函数，绑定一个 DataLogger 对象用于记录数据。
   * @param logger 用于写入数据的 DataLogger 引用
   */
  explicit OffsetMetric(DataLogger &logger);

  /**
   * @brief 记录一组偏移量数据。
   * 
   * @param raw_dx 原始 X 方向偏移量
   * @param raw_dy 原始 Y 方向偏移量
   * @param filtered_dx 滤波后的 X 方向偏移量
   * @param filtered_dy 滤波后的 Y 方向偏移量
   */
  void record(int raw_dx, int raw_dy, float filtered_dx, float filtered_dy);

private:
  DataLogger &logger_;  ///< 用于写入偏移量数据的 DataLogger 引用
};

#endif // FACE_TRACK_GIMBAL_FACE_TRACK_CTRL_TELEMETRY_OFFSET_METRIC_H_
