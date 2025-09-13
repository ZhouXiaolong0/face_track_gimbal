#ifndef FACE_TRACK_GIMBAL_FACE_TRACK_CTRL_CAMERA_H_
#define FACE_TRACK_GIMBAL_FACE_TRACK_CTRL_CAMERA_H_

#include <opencv2/opencv.hpp>

/**
 * @class Camera
 * @brief 摄像头操作类，封装摄像头打开、捕获帧和关闭功能
 *
 * @note 线程安全需用户自行保证
 */
class Camera {
public:
  /**
   * @brief 构造函数
   */
  Camera();

  /**
   * @brief 构造函数：打开指定摄像头并设置分辨率和帧率
   *
   * 此构造函数会立即尝试打开给定 ID 的摄像头，并设置目标分辨率与帧率。
   * 如果打开失败，将抛出 std::runtime_error 异常。
   *
   * @param device_id    摄像头设备 ID（例如 0 表示 /dev/video0）
   * @param width       目标分辨率宽度
   * @param height      目标分辨率高度
   * @param fps         目标帧率
   */
  Camera(int device_id, int width, int height, int fps);

  /**
   * @brief 析构函数，自动关闭摄像头
   */
  ~Camera();

  /**
   * @brief 禁止拷贝构造
   */
  Camera(const Camera &) = delete;

  /**
   * @brief 禁止拷贝赋值
   */
  Camera &operator=(const Camera &) = delete;

  /**
   * @brief 支持移动构造
   */
  Camera(Camera &&) noexcept = default;

  /**
   * @brief 支持移动赋值
   */
  Camera &operator=(Camera &&) noexcept = default;

  // ===== Getter / Setter =====

  /**
   * @brief 判断摄像头是否打开
   *
   * @return true 表示摄像头已打开
   * @return false 表示摄像头未打开
   */
  bool isOpen() const { return cap_.isOpened(); }

  /**
   * @brief 获取摄像头输出宽度设置值
   * @return 摄像头输出宽度设置值
   */
  int getWidth() const { return width_; }

  /**
   * @brief 获取摄像头输出高度设置值
   * @return 摄像头输出高度设置值
   */
  int getHeight() const { return height_; }

  /**
   * @brief 捕获摄像头获取的帧率值
   * @return 帧率值
   */
  int getFPS() const { return fps_; }

  /**
   * @brief 获取 OpenCV 内部属性
   * @param prop_id OpenCV VideoCapture 属性 ID（如
   * CAP_PROP_BRIGHTNESS、CAP_PROP_CONTRAST 等）
   * @return 对应属性值
   */
  double getProperty(int prop_id) const { return cap_.get(prop_id); }

  /**
   * @brief 设置分辨率
   * @param width    图像宽度（像素），默认 640
   * @param height   图像高度（像素），默认 480
   */
  void setResolution(int width = 640, int height = 480);

  /**
   * @brief 设置帧率
   * @param fps      帧率（Frames Per Second），默认 30
   */
  void setFPS(int fps = 30);

  // ===== 核心操作函数 =====

  /**
   * @brief 打开摄像头
   * @param device_id 摄像头设备编号（通常为 0 表示默认摄像头）
   * @param width    图像宽度（像素），默认 640
   * @param height   图像高度（像素），默认 480
   * @param fps      帧率（Frames Per Second），默认 30
   * @return true 打开成功
   * @return false 打开失败
   * @note 使用完毕后应调用 close() 关闭摄像头
   */
  bool open(int device_id, int width = 640, int height = 480, int fps = 30);

  /**
   * @brief 捕获一帧图像
   * @return 返回当前捕获的图像
   * @note 调用此函数前需要确保摄像头已成功打开
   */
  cv::Mat captureFrame();

  /**
   * @brief 关闭摄像头
   */
  void close();

private:
  cv::VideoCapture cap_; /**< OpenCV 摄像头对象 */

  int width_ = 640;  /**< 当前分辨率宽度 */
  int height_ = 480; /**< 当前分辨率高度 */
  int fps_ = 30;     /**< 当前帧率 */
};

#endif // FACE_TRACK_GIMBAL_FACE_TRACK_CTRL_CAMERA_H_