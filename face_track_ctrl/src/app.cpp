#include "app.h"
#include "app_threads.h"
#include "fps_counter.h"

App::App()
    : cfg_(Config::getInstance()), yolo_(nullptr), camera_(nullptr),
      visualizer_(nullptr) {}

App::~App() { cv::destroyAllWindows(); }

void App::initCuda(int cuda_device_id) {

  // 输出 OpenCV 版本
  std::cout << "OpenCV version: " << CV_VERSION << std::endl;

  int numCudaDevices = cv::cuda::getCudaEnabledDeviceCount();
  if (numCudaDevices == 0) {
    throw std::runtime_error("没有检测到支持 CUDA 的设备");
  }

  // 打印所有设备信息
  for (int i = 0; i < numCudaDevices; i++) {
    cv::cuda::DeviceInfo devInfo(i);
    std::cout << "设备 " << i << ": " << devInfo.name() << " | 计算能力 "
              << devInfo.majorVersion() << "." << devInfo.minorVersion()
              << " | 多处理器数 " << devInfo.multiProcessorCount() << std::endl;
  }

  cv::cuda::setDevice(cuda_device_id);

  // 确认当前设备
  cv::cuda::DeviceInfo curDevice(cv::cuda::getDevice());
  std::cout << "当前 CUDA 设备: " << curDevice.name() << std::endl;
}

void App::initYolo() {
  yolo_ = std::make_unique<YoloOnnxRuntime>(cfg_.model_path, cfg_.use_cuda,
                                            cfg_.use_tensorrt);
  yolo_->printAvailableProviders();
}

void App::initCamera() {
  camera_ = std::make_unique<Camera>(cfg_.cam_id, cfg_.cam_width,
                                     cfg_.cam_height, cfg_.cam_fps);
}

void App::initVisualizer() {
  visualizer_ = std::make_unique<Visualizer>(cfg_.cam_width, cfg_.cam_height);
}

void App::initSerialPort() {
  serial_ =
      std::make_unique<SerialPort>(cfg_.serial_device, cfg_.serial_baudrate);
}

void App::run() {
  initCuda(cfg_.cuda_device_id);
  initYolo();
  initCamera();
  initVisualizer();
  initSerialPort();

  AppThreads threads(camera_.get(), yolo_.get(), visualizer_.get(),
                     serial_.get(), cfg_);

  threads.start();

  // ----------------
  // 显示线程（主线程）
  // ----------------
  FPSCounter fpsCounter(cfg_.fps_window_size); // 窗口大小 30 帧

  while (true) {
    auto res = threads.popResult();
    auto &frame = res.first;
    auto &detRes = res.second;

    // 更新 FPS
    fpsCounter.tick();
    double fps = fpsCounter.getFPS();

    visualizer_->renderFrame(frame, detRes.rects, detRes.confs, detRes.indices,
                             fps);
    cv::imshow("ONNX YOLOv5su", frame);

    // 把检测结果（不含 frame） + 当前帧宽高 推到 det_res_queue
    DetectionPayload payload;
    payload.det = detRes; // 拷贝 DetectionResult（通常较小）
    payload.frame_width = frame.cols;
    payload.frame_height = frame.rows;
    threads.pushDetectionPayload(payload);

    int key = cv::waitKey(1);
    if (key == 27) { // ESC 退出
      break;
    }
  }

  threads.stop();
}