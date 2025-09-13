#ifndef FACE_TRACK_GIMBAL_FACE_TRACK_CTRL_SERIAL_PORT_H_
#define FACE_TRACK_GIMBAL_FACE_TRACK_CTRL_SERIAL_PORT_H_

#include <fcntl.h>
#include <iostream>
#include <sstream>
#include <string>
#include <termios.h>
#include <unistd.h>

/**
 * @brief 封装串口通信操作的类
 *
 * 提供打开串口、发送偏移量数据以及关闭串口等功能。
 */
class SerialPort {

public:
  /**
   * @brief 构造函数，初始化串口对象但不打开串口
   *
   * @param device 串口设备路径，例如 "/dev/ttyUSB0"
   * @param baudrate 波特率，例如 9600、115200
   */
  SerialPort(const std::string &device, int baudrate);

  /**
   * @brief 析构函数，自动关闭串口
   *
   */
  ~SerialPort();

  /**
   * @brief 检查串口是否已经打开
   *
   * @return true 串口已打开
   * @return false 串口未打开
   */
  bool isOpen() const;

  /**
   * @brief 打开指定串口
   *
   * @param device 串口设备路径
   * @param baudrate 波特率
   * @return true 打开成功
   * @return false 打开失败
   */
  bool openPort(const std::string &device, int baudrate);

  /**
   * @brief 发送偏移量数据到串口
   *
   * 一般用于控制云台或者其他设备的偏移量指令。
   *
   * @param dx x 方向偏移
   * @param dy y 方向偏移
   * @return true 发送成功
   * @return false 发送失败
   */
  bool sendOffsets(int dx, int dy);

  /**
   * @brief 关闭串口
   *
   */
  void closePort();

private:
  int fd_; ///< 串口文件描述符
};

#endif // FACE_TRACK_GIMBAL_FACE_TRACK_CTRL_SERIAL_PORT_H_
