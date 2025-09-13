#ifndef FACE_TRACK_GIMBAL_FACE_TRACK_CTRL_CONFIG_H_
#define FACE_TRACK_GIMBAL_FACE_TRACK_CTRL_CONFIG_H_

#include <string>

/**
 * @brief 全局配置类（单例模式）。
 *
 * 该类用于集中管理应用程序的所有配置参数，包括摄像头、
 * CUDA、模型、推理、可视化、云台控制以及串口相关的设置。
 *
 * 使用方式：
 * @code
 *   Config& cfg = Config::getInstance();
 *   int camId = cfg.camId;
 * @endcode
 */
class Config {
public:
  /**
   * @brief 获取配置单例实例。
   *
   * @return Config& 单例引用。
   */
  static Config &getInstance();

  /// 禁止拷贝构造
  Config(const Config &) = delete;

  /// 禁止赋值操作
  Config &operator=(const Config &) = delete;

  // ---------------------
  // 摄像头相关
  // ---------------------
  int cam_id;     ///< 摄像头设备号
  int cam_width;  ///< 摄像头分辨率宽度
  int cam_height; ///< 摄像头分辨率高度
  int cam_fps;    ///< 摄像头帧率

  // ---------------------
  // CUDA 相关
  // ---------------------
  int cuda_device_id; ///< CUDA 设备 ID，默认 0 表示 GPU0

  // ---------------------
  // 模型相关
  // ---------------------
  std::string model_path; ///< ONNX 模型路径
  bool use_cuda;          ///< 是否启用 CUDA 加速
  bool use_tensorrt;      ///< 是否启用 TensorRT 加速

  // ---------------------
  // 推理相关
  // ---------------------
  float conf_threshold; ///< 置信度阈值
  float iou_threshold;  ///< NMS IOU 阈值
  int input_width;      ///< 模型输入宽度
  int input_height;     ///< 模型输入高度

  // ---------------------
  // 可视化相关
  // ---------------------
  bool draw_center_cross; ///< 是否绘制图像中心十字

  // ---------------------
  // 云台控制相关
  // ---------------------
  int gimbal_threshold; ///< 云台移动偏移量阈值

  // ---------------------
  // 串口相关
  // ---------------------
  std::string serial_device; ///< 串口设备名（如 /dev/ttyUSB0）
  int serial_baudrate;       ///< 串口波特率（如 115200）

  // ---------------------
  // fps 计算相关
  // ---------------------
  int fps_window_size; ///< 计算 fps 窗口大小

  // ---------------------
  // 扩展接口（以后用）
  // ---------------------
  /**
   * @brief 从配置文件加载参数。
   * @param path 配置文件路径。
   * @return true 加载成功。
   * @return false 加载失败。
   */
  bool loadFromFile(const std::string &path);

  /**
   * @brief 将当前配置保存到文件。
   * @param path 输出文件路径。
   * @return true 保存成功。
   * @return false 保存失败。
   */
  bool saveToFile(const std::string &path) const;

private:
  /**
   * @brief 构造函数（私有化，单例模式）。
   *
   */
  Config(); // 构造函数私有化
};

#endif // FACE_TRACK_GIMBAL_FACE_TRACK_CTRL_CONFIG_H_
