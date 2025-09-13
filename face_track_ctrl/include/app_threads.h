#ifndef FACE_TRACK_GIMBAL_FACE_TRACK_CTRL_APP_THREADS_H_
#define FACE_TRACK_GIMBAL_FACE_TRACK_CTRL_APP_THREADS_H_

#include "camera.h"
#include "config.h"
#include "gimbal_controller.h"
#include "serial_port.h"
#include "telemetry/system_metric.h"
#include "thread_safe_queue.h"
#include "visualizer.h"
#include "yolo_onnx_runtime.h"
#include <atomic>
#include <opencv2/opencv.hpp>
#include <thread>
#include <vector>

/**
 * @brief 封装检测结果及图像尺寸信息的结构体
 */
struct DetectionPayload {
  DetectionResult det;
  int frame_width;
  int frame_height;
};

/**
 * @brief 管理应用中所有线程的类（采集、推理、云台控制、系统日志）
 *
 * AppThreads 类封装了整个系统运行过程中所需的多线程任务：
 * - 图像采集
 * - 推理（YOLO）
 * - 偏移量计算
 * - 云台控制
 * - 系统指标日志
 *
 * 线程间通过线程安全队列进行数据交换。
 */
class AppThreads {
public:
  /**
   * @brief 构造函数
   * @param cam 相机对象指针
   * @param yolo YOLO 推理引擎指针
   * @param vis 可视化器对象指针
   * @param serial 串口对象指针
   * @param cfg 系统配置
   */
  explicit AppThreads(Camera *cam, YoloOnnxRuntime *yolo, Visualizer *vis,
                      SerialPort *serial, const Config &cfg);

  /**
   * @brief 析构函数，确保线程安全退出
   */
  ~AppThreads();

  /**
   * @brief 启动所有线程
   */
  void start();

  /**
   * @brief 停止所有线程
   */
  void stop();

  /**
   * @brief 阻塞获取推理结果
   * @return 包含图像和检测结果的 pair
   */
  std::pair<cv::Mat, DetectionResult> popResult() {
    return result_queue_.pop(); // 阻塞直到有数据
  }

  /**
   * @brief 阻塞获取检测结果负载
   * @return DetectionPayload 对象
   */
  DetectionPayload popDetectionPayload() { return det_res_queue_.pop(); }

  /**
   * @brief 推送检测结果负载到队列
   * @param payload 检测结果负载
   */
  void pushDetectionPayload(const DetectionPayload &payload) {
    det_res_queue_.push(payload);
  }

private:
  /**
   * @brief 图像采集线程循环
   */
  void captureLoop();

  /**
   * @brief 推理线程循环
   */
  void inferenceLoop();

  /**
   * @brief 偏移量计算线程循环
   */
  void offsetCalcLoop();

  /**
   * @brief 云台控制线程循环
   */
  void gimbalLoop();

  /**
   * @brief 系统指标记录线程循环
   */
  void systemLoggerLoop();

  Camera *camera_;
  YoloOnnxRuntime *yolo_;
  Visualizer *visualizer_;
  SerialPort *serial_;
  const Config &cfg_;

  std::atomic<bool> running_{false};

  ThreadSafeQueue<cv::Mat> capture_queue_;
  ThreadSafeQueue<GimbalController::Offset> offset_queue_;
  ThreadSafeQueue<std::pair<cv::Mat, DetectionResult>> result_queue_;
  ThreadSafeQueue<DetectionPayload> det_res_queue_;

  std::vector<std::thread> threads_;
};

#endif // FACE_TRACK_GIMBAL_FACE_TRACK_CTRL_APP_THREADS_H_