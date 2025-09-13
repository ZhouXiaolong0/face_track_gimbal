#ifndef FACE_TRACK_GIMBAL_FACE_TRACK_CTRL_YOLO_ONNX_RUNTIME_H_
#define FACE_TRACK_GIMBAL_FACE_TRACK_CTRL_YOLO_ONNX_RUNTIME_H_

#include "frame_processor.h"

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

/**
 * @brief 解码结果结构体
 *
 * 存储网络输出解码后的边界框和对应的置信度
 */
struct DecodeResult {
  std::vector<cv::Vec4f> boxes; /**< [x, y, w, h] 格式的边界框 */
  std::vector<float> scores;    /**< 对应边界框的置信度 */
};

/**
 * @brief 检测结果结构体
 *
 * 存储最终映射到原图的矩形框、置信度和 NMS 保留索引
 */
struct DetectionResult {
  std::vector<cv::Rect> rects; /**< 最终映射到原图并整型化的矩形 */
  std::vector<float> confs;    /**< 对应矩形的置信度 */
  std::vector<int> indices;    /**< NMS 保留下来的索引（可选） */
};

/**
 * @brief YOLO ONNX Runtime 封装类
 *
 * 提供 ONNX Runtime 模型加载、推理、解码、NMS 以及端到端检测功能
 */
class YoloOnnxRuntime {
public:
  /**
   * @brief 构造函数
   *
   * @param model_path  模型路径
   * @param use_cuda  是否使用 CUDA
   * @param use_tensorRT  是否使用 TensorRT
   * @param intra_op_num_threads CPU 线程数
   */
  explicit YoloOnnxRuntime(const std::string &model_path, bool use_cuda = true,
                           bool use_tensorRT = true,
                           int intra_op_num_threads = 4);

  /**
   * @brief 打印可用的执行提供器 (Execution Provider)
   *
   */
  void printAvailableProviders();

  /**
   * @brief 获取输入张量名称
   *
   * @return 输入张量名称列表
   */
  const std::vector<const char *> &getInputNames() const {
    return input_names_;
  }

  /**
   * @brief 获取输出张量名称
   *
   * @return 输出张量名称列表
   */
  const std::vector<const char *> &getOutputNames() const {
    return output_names_;
  }

  /**
   * @brief 获取 ONNX Runtime Session
   *
   * @return Session 引用
   */
  Ort::Session &getSession() { return *session_; }

  /**
   * @brief 解码输出张量
   *
   * @param output_tensor 网络输出张量
   * @param conf_thres 置信度阈值
   * @return DecodeResult 解码后的边界框和置信度
   */
  DecodeResult decode(const Ort::Value &output_tensor, float conf_thres = 0.3);

  /**
   * @brief 对预处理后的图像进行推理并解码
   *
   * @param processed_frame 已预处理的图像 (HWC -> CHW, 归一化)
   * @param conf_thres 置信度阈值
   * @return DecodeResult 解码后的边界框和置信度
   */
  DecodeResult infer(const cv::Mat &processed_frame, float conf_thres = 0.3);

  /**
   * @brief 端到端检测：infer + xywh->xyxy + map -> nms
   *
   * @param processed_frame 已预处理的图像
   * @param meta 预处理元信息，用于映射回原图
   * @param conf_thres 置信度阈值
   * @param iou_thres NMS IOU 阈值
   * @return DetectionResult 最终检测结果
   */
  DetectionResult detect(const cv::Mat &processed_frame,
                         const FrameProcessor::TransformMeta &meta,
                         float conf_thres = 0.3f, float iou_thres = 0.45f);

private:
  Ort::Env env_;                               /**< ONNX Runtime 环境 */
  Ort::SessionOptions session_options_;        /**< Session 配置选项 */
  std::unique_ptr<Ort::Session> session_;      /**< Session 指针 */
  Ort::AllocatorWithDefaultOptions allocator_; /**< 默认分配器 */

  std::vector<const char *> input_names_;  /**< 输入张量名称 */
  std::vector<const char *> output_names_; /**< 输出张量名称 */

  /**
   * @brief 添加 CUDA Execution Provider
   *
   * @param cuda_device_id GPU 设备 ID
   * @param gpu_mem_limit GPU 内存限制
   * @param arena_extend_strategy 内存分配策略
   * @param do_copy_in_default_stream 是否使用默认流拷贝
   * @param cudnn_algo_search 卷积算法搜索策略
   */
  void appendCudaProvider(int cuda_device_id = 0,
                          size_t gpu_mem_limit = SIZE_MAX,
                          int arena_extend_strategy = 0,
                          bool do_copy_in_default_stream = true,
                          OrtCudnnConvAlgoSearch cudnn_algo_search =
                              OrtCudnnConvAlgoSearchExhaustive);

  /**
   * @brief 添加 TensorRT Execution Provider
   *
   * @param max_workspace_size TensorRT 最大工作空间
   * @param fp16_enable 是否启用 FP16
   */
  void appendTensorRTProvider(size_t max_workspace_size = 256 * 1024 * 1024,
                              bool fp16_enable = false);

  /**
   * @brief 加载模型
   *
   * @param model_path 模型路径
   */
  void loadModel(const std::string &model_path);
};

#endif // FACE_TRACK_GIMBAL_FACE_TRACK_CTRL_YOLO_ONNX_RUNTIME_H_