#include "serial_port.h"

SerialPort::SerialPort(const std::string &device, int baudrate) : fd_(-1) {
  if (!openPort(device, baudrate)) {
    std::cerr << "串口打开失败: " << device << std::endl;
  }
}

SerialPort::~SerialPort() { closePort(); }

bool SerialPort::isOpen() const { return fd_ >= 0; }

bool SerialPort::openPort(const std::string &device, int baudrate) {
  fd_ = open(device.c_str(), O_RDWR | O_NOCTTY | O_SYNC);
  if (fd_ < 0) {
    perror("open");
    return false;
  }

  struct termios tty;
  if (tcgetattr(fd_, &tty) != 0) {
    perror("tcgetattr");
    close(fd_);
    fd_ = -1;
    return false;
  }

  cfmakeraw(&tty);

  speed_t speed;
  switch (baudrate) {
  case 9600:
    speed = B9600;
    break;
  case 19200:
    speed = B19200;
    break;
  case 38400:
    speed = B38400;
    break;
  case 57600:
    speed = B57600;
    break;
  case 115200:
    speed = B115200;
    break;
  default:
    speed = B115200;
    break;
  }
  cfsetospeed(&tty, speed);
  cfsetispeed(&tty, speed);

  tty.c_cflag |= (CLOCAL | CREAD);
  tty.c_cflag &= ~CSTOPB;
  tty.c_cflag &= ~PARENB;
  tty.c_cflag &= ~CSIZE;
  tty.c_cflag |= CS8;

  tty.c_cc[VMIN] = 0;
  tty.c_cc[VTIME] = 10;

  tcflush(fd_, TCIFLUSH);
  if (tcsetattr(fd_, TCSANOW, &tty) != 0) {
    perror("tcsetattr");
    close(fd_);
    fd_ = -1;
    return false;
  }

  return true;
}

bool SerialPort::sendOffsets(int dx, int dy) {
  if (fd_ < 0)
    return false;

  std::ostringstream ss;
  ss << dx << "," << dy << "\n";
  std::string s = ss.str();

  ssize_t n = write(fd_, s.c_str(), s.size());
  if (n != (ssize_t)s.size()) {
    perror("write");
    return false;
  }

  tcdrain(fd_);
  return true;
}

void SerialPort::closePort() {
  if (fd_ >= 0) {
    close(fd_);
    fd_ = -1;
  }
}