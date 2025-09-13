#include "config.h"
#include <iostream>

Config &Config::getInstance() {
  static Config instance;
  return instance;
}

Config::Config()
    : cam_id(0), cam_width(640), cam_height(480), cam_fps(30),
      cuda_device_id(0), model_path("../models/yolov5su.onnx"), use_cuda(true),
      use_tensorrt(true), conf_threshold(0.3f), iou_threshold(0.45f),
      input_width(640), input_height(640), draw_center_cross(true),
      gimbal_threshold(20), serial_device("/dev/ttyTHS1"),
      serial_baudrate(115200), fps_window_size(30) {}

bool Config::loadFromFile(const std::string &path) {
  // 预留：这里可以扩展成 JSON / YAML 配置读取
  std::cout << "[Config] 加载配置文件: " << path << std::endl;
  return true;
}

bool Config::saveToFile(const std::string &path) const {
  // 预留：这里可以扩展成 JSON / YAML 配置保存
  std::cout << "[Config] 保存配置文件到: " << path << std::endl;
  return true;
}
