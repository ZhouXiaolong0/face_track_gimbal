#include "app_threads.h"
#include "config.h"
#include "frame_processor.h"
#include "telemetry/system_info.h"
#include <chrono>
#include <iostream>

AppThreads::AppThreads(Camera *cam, YoloOnnxRuntime *yolo, Visualizer *vis,
                       SerialPort *serial, const Config &cfg)
    : camera_(cam), yolo_(yolo), visualizer_(vis), serial_(serial), cfg_(cfg),
      capture_queue_(5), offset_queue_(5), result_queue_(5), det_res_queue_(5) {
}

AppThreads::~AppThreads() { stop(); }

void AppThreads::start() {
  running_ = true;
  threads_.emplace_back(&AppThreads::captureLoop, this);
  threads_.emplace_back(&AppThreads::inferenceLoop, this);
  threads_.emplace_back(&AppThreads::gimbalLoop, this);
  threads_.emplace_back(&AppThreads::offsetCalcLoop, this);
  threads_.emplace_back(&AppThreads::systemLoggerLoop, this);
}

void AppThreads::stop() {
  running_ = false;
  for (auto &t : threads_) {
    if (t.joinable())
      t.join();
  }
}

// ----------------
// 采集线程
// ----------------
void AppThreads::captureLoop() {
  while (running_) {
    auto frame = camera_->captureFrame();
    if (!frame.empty())
      capture_queue_.push(frame);
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
}

// ----------------
// 推理线程
// ----------------
void AppThreads::inferenceLoop() {
  while (running_) {
    auto frame = capture_queue_.pop();
    auto proc = FrameProcessor::processFrame(frame);
    // auto start = std::chrono::high_resolution_clock::now();
    auto detRes = yolo_->detect(proc.processed_frame, proc.meta,
                                cfg_.conf_threshold, cfg_.iou_threshold);
    // auto end = std::chrono::high_resolution_clock::now();
    // std::cout
    //     << "Inference time: "
    //     << std::chrono::duration<double, std::milli>(end - start).count()
    //     << " ms" << std::endl;

    result_queue_.push({frame, detRes});

    // 打印队列大小，确认有内容
    // size_t qSize = result_queue.size();
    // std::cout << "[InferenceThread] result_queue size: " << qSize <<
    // std::endl;
  }
}

// ----------------
// 新增：偏移量计算线程（从 det_res_queue_ 中 pop，计算 offset，push 到
// offset_queue_）
// ----------------
void AppThreads::offsetCalcLoop() {
  while (running_) {
    auto payload = det_res_queue_.pop();
    const auto &detRes = payload.det;
    int w = payload.frame_width;
    int h = payload.frame_height;

    for (int idx : detRes.indices) {
      cv::Rect r = detRes.rects[idx];
      auto offset = GimbalController::computeOffsets(r, w, h);
      if (!GimbalController::needMove(offset, cfg_.gimbal_threshold)) {
        // 太小就跳过
        std::cout << "Difference is too small, no need to move the servo"
                  << std::endl;
        continue;
      }

      std::cout << "dx is :" << offset.dx << "dy is :" << offset.dy
                << std::endl;

      // 推到已有的 offsetQueue，由 gimbal_thread 负责发送
      offset_queue_.push(offset);
    }
  }
}

// ----------------
// 发送偏移量线程
// ----------------
void AppThreads::gimbalLoop() {
  while (running_) {
    auto offset = offset_queue_.pop();
    if (serial_->isOpen()) {
      serial_->sendOffsets(offset.dx, offset.dy);
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
}

// ----------------
// 系统指标记录线程
// ----------------
void AppThreads::systemLoggerLoop() {
  DataLogger sys_logger("system_info.csv");
  SystemMetric sys_metric(sys_logger);
  while (running_) {
    auto info = SystemInfo::collect();
    sys_metric.record(info.cpu, info.mem, info.gpu);
    std::this_thread::sleep_for(std::chrono::seconds(1));
  }
}
