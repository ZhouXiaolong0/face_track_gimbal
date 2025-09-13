#include "camera.h"
#include <iostream>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>

Camera::Camera() {}

Camera::Camera(int device_id, int width, int height, int fps) {
  if (!open(device_id, width, height, fps)) {
    throw std::runtime_error(
        "Failed to open camera in constructor (device_id=" +
        std::to_string(device_id) + ")");
  }
}

Camera::~Camera() { close(); }

void Camera::setResolution(int width, int height) {
  if (cap_.isOpened()) {
    cap_.set(cv::CAP_PROP_FRAME_WIDTH, width);
    cap_.set(cv::CAP_PROP_FRAME_HEIGHT, height);
  }
}

void Camera::setFPS(int fps) {
  if (cap_.isOpened()) {
    cap_.set(cv::CAP_PROP_FPS, fps);
  }
}

bool Camera::open(int device_id, int width, int height, int fps) {
  cap_.open(device_id);
  if (!cap_.isOpened()) {
    std::cerr << "Failed to open camera ID " << device_id << std::endl;
    return false;
  }

  this->setResolution(width, height);
  this->setFPS(fps);

  return true;
}

cv::Mat Camera::captureFrame() {
  if (!cap_.isOpened()) {
    // 摄像头未打开，返回空帧
    return cv::Mat();
  }

  cv::Mat frame;
  cap_ >> frame;

  // 捕获失败返回空帧并打印警告
  if (frame.empty()) {
    std::cerr << "[Camera] Warning: captured empty frame" << std::endl;
  }

  return frame;
}

void Camera::close() {
  if (cap_.isOpened()) {
    cap_.release();
  }
}