#ifndef FACE_TRACK_GIMBAL_FACE_TRACK_CTRL_APP_H_
#define FACE_TRACK_GIMBAL_FACE_TRACK_CTRL_APP_H_

#include "camera.h"
#include "config.h"
#include "frame_processor.h"
#include "serial_port.h"
#include "visualizer.h"
#include "yolo_onnx_runtime.h"

/**
 * @brief 应用程序的主控制类。
 *
 * 该类负责初始化和管理 YOLO 推理、摄像头采集、
 * 可视化模块等组件，并提供统一的运行入口。
 */
class Camera;
class SerialPort;
class Visualizer;
class YoloOnnxRuntime;

/**
 * @brief 应用程序的主控制类。
 *
 * 该类负责初始化和管理 YOLO 推理、摄像头采集、
 * 可视化模块等组件，并提供统一的运行入口。
 */
class App {
public:
  /**
   * @brief 构造函数。
   */
  explicit App();

  /**
   * @brief 析构函数。
   */
  ~App();

  /**
   * @brief 程序运行入口。
   *
   * 该函数会启动摄像头采集、YOLO 推理和可视化流程。
   */
  void run();

private:
  /**
   * @brief 初始化 CUDA 环境。
   *
   * @param cuda_device_id 使用的 CUDA 设备 ID，默认值为 0。
   */
  void initCuda(int cuda_device_id = 0);

  /**
   * @brief 初始化 YOLO 推理引擎。
   *
   */
  void initYolo();

  /**
   * @brief 初始化摄像头模块。
   *
   */
  void initCamera();

  /**
   * @brief 初始化可视化模块。
   *
   */
  void initVisualizer();

  /**
   * @brief 串口初始化模块。
   *
   */
  void initSerialPort();

  // 配置
  Config &cfg_;

  std::unique_ptr<YoloOnnxRuntime> yolo_;  /**< YOLO 推理引擎实例。 */
  std::unique_ptr<Camera> camera_;         /**< 摄像头采集实例。 */
  std::unique_ptr<Visualizer> visualizer_; /**< 可视化显示实例。 */
  std::unique_ptr<SerialPort> serial_;     /**< 串口实例。 */
};

#endif // FACE_TRACK_GIMBAL_FACE_TRACK_CTRL_APP_H_