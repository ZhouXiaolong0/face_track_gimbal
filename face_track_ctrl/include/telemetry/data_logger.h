#ifndef FACE_TRACK_GIMBAL_FACE_TRACK_CTRL_TELEMETRY_DATA_LOGGER_H_
#define FACE_TRACK_GIMBAL_FACE_TRACK_CTRL_TELEMETRY_DATA_LOGGER_H_

#include <fstream>
#include <string>
#include <vector>

/**
 * @brief 数据记录器类，用于将数据写入 CSV 或文本文件。
 */
class DataLogger {
public:
  /**
   * @brief 构造函数，打开指定文件用于记录数据。
   * @param filename 要写入的文件名
   */
  explicit DataLogger(const std::string &filename);

  /**
   * @brief 析构函数，关闭文件流。
   */
  ~DataLogger();

  /**
   * @brief 将一行数据写入文件。
   * @param row 要写入的一行数据，每个元素对应一列
   */
  void log(const std::vector<std::string> &row);

private:
  std::ofstream file_;
};

#endif // FACE_TRACK_GIMBAL_FACE_TRACK_CTRL_TELEMETRY_DATA_LOGGER_H_