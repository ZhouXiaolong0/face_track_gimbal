#ifndef FACE_TRACK_GIMBAL_FACE_TRACK_CTRL_TELEMETRY_SYSTEM_METRIC_H_
#define FACE_TRACK_GIMBAL_FACE_TRACK_CTRL_TELEMETRY_SYSTEM_METRIC_H_

#include "data_logger.h"

/**
 * @brief 系统性能度量类，用于记录 CPU、内存和 GPU 使用情况。
 * 
 * 使用 DataLogger 将系统性能数据写入文件，便于分析和调试。
 */
class SystemMetric {
public:
  /**
   * @brief 构造函数，绑定一个 DataLogger 对象用于记录数据。
   * @param logger 用于写入数据的 DataLogger 引用
   */
  explicit SystemMetric(DataLogger &logger);

  /**
   * @brief 记录一组系统性能数据。
   * 
   * @param cpu CPU 使用率（百分比）
   * @param mem 内存使用率（百分比）
   * @param gpu GPU 使用率（百分比）
   */
  void record(float cpu, float mem, float gpu);

private:
  DataLogger &logger_; ///< 用于写入系统性能数据的 DataLogger 引用
};

#endif // FACE_TRACK_GIMBAL_FACE_TRACK_CTRL_TELEMETRY_SYSTEM_METRIC_H_