// data_logger.cpp
#include "telemetry/data_logger.h"
#include <sstream>
#include <stdexcept>

DataLogger::DataLogger(const std::string &filename) {
  file_.open(filename, std::ios::out | std::ios::app);
  if (!file_.is_open()) {
    throw std::runtime_error("无法打开日志文件: " + filename);
  }
}

DataLogger::~DataLogger() {
  if (file_.is_open()) {
    file_.close();
  }
}

void DataLogger::log(const std::vector<std::string> &row) {
  for (size_t i = 0; i < row.size(); ++i) {
    file_ << row[i];
    if (i + 1 < row.size())
      file_ << ",";
  }
  file_ << "\n";

  // 立即将缓冲区内容写入文件
  file_.flush();
}
