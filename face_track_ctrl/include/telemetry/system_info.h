#ifndef FACE_TRACK_GIMBAL_FACE_TRACK_CTRL_TELEMETRY_SYSTEM_INFO_H_
#define FACE_TRACK_GIMBAL_FACE_TRACK_CTRL_TELEMETRY_SYSTEM_INFO_H_

/**
 * @brief 系统状态数据结构，保存 CPU、内存、GPU 使用率。
 */
struct SystemInfoData {
  float cpu;
  float mem;
  float gpu;
};

/**
 * @brief 提供获取系统状态的接口 (CPU、内存、GPU 使用率)。
 *
 * CPU / 内存 从 /proc 文件系统读取。
 * GPU 目前返回 0.0，可以根据需要扩展 (NVML / tegrastats)。
 */
class SystemInfo {
public:
  static float getCpuUsage();
  static float getMemUsage();
  static float getGpuUsage(); // 目前占位
  static SystemInfoData collect();
};

#endif // FACE_TRACK_GIMBAL_FACE_TRACK_CTRL_TELEMETRY_SYSTEM_INFO_H_
