#include "yolo_onnx_runtime.h"
#include <iostream>

YoloOnnxRuntime::YoloOnnxRuntime(const std::string &model_path, bool use_cuda,
                                 bool use_tensorRT, int intra_op_num_threads)
    : env_(ORT_LOGGING_LEVEL_WARNING, "YoloOnnxRuntime") {
  // ORT_API_VERSION: 11
  std::cout << "ORT_API_VERSION: " << ORT_API_VERSION << std::endl;

  // 配置 Session Options
  session_options_.SetIntraOpNumThreads(intra_op_num_threads);

  if (use_cuda) {
    appendCudaProvider();
  }
  if (use_tensorRT) {
    appendTensorRTProvider();
  }

  // 加载模型
  loadModel(model_path);

  // 解析输入输出名
  size_t numInputs = session_->GetInputCount();
  size_t numOutputs = session_->GetOutputCount();

  for (size_t i = 0; i < numInputs; i++) {
    input_names_.push_back(session_->GetInputName(i, allocator_));
  }
  for (size_t i = 0; i < numOutputs; i++) {
    output_names_.push_back(session_->GetOutputName(i, allocator_));
  }
}

// 打印可用的 execute provider
void YoloOnnxRuntime::printAvailableProviders() {
  int provider_count = 0;
  char **provider_names = nullptr;

  Ort::ThrowOnError(
      Ort::GetApi().GetAvailableProviders(&provider_names, &provider_count));

  std::cout << "Available execution providers:" << std::endl;
  for (int i = 0; i < provider_count; i++) {
    std::cout << "  - " << provider_names[i] << std::endl;
  }

  // 释放内存
  Ort::GetApi().ReleaseAvailableProviders(provider_names, provider_count);
}

// 配置 CUDA Execution provider
void YoloOnnxRuntime::appendCudaProvider(
    int cuda_device_id, size_t gpu_mem_limit, int arena_extend_strategy,
    bool do_copy_in_default_stream, OrtCudnnConvAlgoSearch cudnn_algo_search) {
  OrtCUDAProviderOptions cuda_options{};
  cuda_options.device_id = cuda_device_id;    // 使用 GPU0
  cuda_options.gpu_mem_limit = gpu_mem_limit; // 使用尽可能多的显存
  cuda_options.arena_extend_strategy = arena_extend_strategy;
  cuda_options.do_copy_in_default_stream = do_copy_in_default_stream ? 1 : 0;
  cuda_options.cudnn_conv_algo_search = cudnn_algo_search;

  session_options_.AppendExecutionProvider_CUDA(cuda_options);
}

// 配置 tensorRT provider
void YoloOnnxRuntime::appendTensorRTProvider(size_t max_workspace_size,
                                             bool fp16_enable) {
  OrtTensorRTProviderOptionsV2 *trt_options = nullptr;
  Ort::ThrowOnError(Ort::GetApi().CreateTensorRTProviderOptions(&trt_options));

  const char *keys[] = {"trt_max_workspace_size", "trt_fp16_enable"};
  std::string workspace_str = std::to_string(max_workspace_size);
  const char *values[] = {workspace_str.c_str(), fp16_enable ? "1" : "0"};

  Ort::ThrowOnError(Ort::GetApi().UpdateTensorRTProviderOptions(
      trt_options, keys, values, 2));

  Ort::ThrowOnError(
      Ort::GetApi().SessionOptionsAppendExecutionProvider_TensorRT_V2(
          session_options_, trt_options));

  Ort::GetApi().ReleaseTensorRTProviderOptions(trt_options);
}

void YoloOnnxRuntime::loadModel(const std::string &model_path) {
  session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(),
                                            session_options_);
}

DecodeResult YoloOnnxRuntime::decode(const Ort::Value &output_tensor,
                                     float conf_thres) {
  DecodeResult result;

  // YOLOv5 输出形状: (1, 84, 8400)
  auto typeInfo = output_tensor.GetTensorTypeAndShapeInfo();
  std::vector<int64_t> shape = typeInfo.GetShape();

  int64_t batch = shape[0];       // 1
  int64_t num_attrs = shape[1];   // 84
  int64_t num_anchors = shape[2]; // 8400

  const float *preds = output_tensor.GetTensorData<float>();

  auto sigmoid = [](float x) { return 1.f / (1.f + std::exp(-x)); };

  for (int i = 0; i < num_anchors; ++i) {
    // xywh
    float cx = preds[0 * num_anchors + i];
    float cy = preds[1 * num_anchors + i];
    float w = preds[2 * num_anchors + i];
    float h = preds[3 * num_anchors + i];

    float objectness = sigmoid(preds[4 * num_anchors + i]);
    float human_logit = preds[5 * num_anchors + i]; // “人”类别（索引=0）
    float human_prob = sigmoid(human_logit);

    float score = objectness * human_prob;

    if (score > conf_thres) {
      result.boxes.push_back(cv::Vec4f(cx, cy, w, h));
      result.scores.push_back(score);
    }
  }

  return result;
}

DecodeResult YoloOnnxRuntime::infer(const cv::Mat &processed_frame,
                                    float conf_thres) {
  // 1. HWC -> CHW
  std::vector<float> input_tensor_values =
      FrameProcessor::toCHW(processed_frame);

  // 2. 定义输入 shape 和 memory_info
  std::vector<int64_t> input_shape = {1, 3, processed_frame.rows,
                                      processed_frame.cols}; // batch=1
  Ort::MemoryInfo memory_info =
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  // 3. 构造 ONNX 输入张量
  Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
      memory_info, input_tensor_values.data(), input_tensor_values.size(),
      input_shape.data(), input_shape.size());

  // 4. 推理
  // output_tensors 形状为 (batch, num_attrs, num_anchors),(1, 84, 8400)
  // batch = 1
  // num_attes = 84 -> 4bbox + 1 objectness + 79 个类别
  // 前四个：边界框
  // 第五个:objectness 置信度
  // 后 79 个：类别，总共 80 类。
  auto output_tensors = session_->Run(
      Ort::RunOptions{nullptr}, input_names_.data(), &input_tensor, 1,
      output_names_.data(), output_names_.size());

  // 获取第一个输出张量
  const Ort::Value &output_tensor = output_tensors[0];

  // 5. decode + confidence filter
  DecodeResult decode_res = this->decode(output_tensor, conf_thres);

  return decode_res;
}

// detect: infer -> xywh->xyxy -> mapBoxesToOriginal -> nms -> 返回
// DetectionResult
DetectionResult
YoloOnnxRuntime::detect(const cv::Mat &processed_frame,
                        const FrameProcessor::TransformMeta &meta,
                        float conf_thres, float iou_thres) {
  DetectionResult result;

  // 1) infer 得到 xywh + scores（infer 内部会用 conf_thres 对 score
  // 做初筛，如果你希望在 nms 前再筛一次可以调整）
  DecodeResult decoded = this->infer(processed_frame, conf_thres);

  // 2) 如果空，直接返回空结果（避免后续越界）
  if (decoded.boxes.empty()) {
    return result;
  }

  // 3）把边界框从 [中心点 x, 中心点 y, 宽, 高] 转换为 [x1, y1, x2, y2]
  // x1, y1 左上角，x2，y2 右下角
  FrameProcessor::xywh2xyxy(decoded.boxes); // 转换成 xyxy

  // 4）映射回原图, boxes: xyxy 格式
  auto mapped_boxes = FrameProcessor::mapBoxesToOriginal(decoded.boxes, meta);

  // 5) NMS（你的 FrameProcessor::nms 接受 xyxy boxes 与 scores）
  FrameProcessor::NMSResult nms_res =
      FrameProcessor::nms(mapped_boxes, decoded.scores, conf_thres, iou_thres);

  // 6) 填充 DetectionResult（将 rects/conf/indices 直接返回）
  result.rects = std::move(nms_res.rects);
  result.confs = std::move(nms_res.confs);
  result.indices = std::move(nms_res.indices);

  return result;
}