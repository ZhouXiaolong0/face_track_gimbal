#ifndef FACE_TRACK_GIMBAL_FACE_TRACK_CTRL_GIMBAL_CONTROLLER_H_
#define FACE_TRACK_GIMBAL_FACE_TRACK_CTRL_GIMBAL_CONTROLLER_H_

#include <opencv2/opencv.hpp>

/**
 * @brief 云台控制器工具类。
 *
 * 该类提供计算目标框相对于图像中心的偏移量，
 * 以及判断是否需要移动云台的静态方法。
 */
class GimbalController {
public:
  /**
   * @brief 偏移量结构体。
   *
   * 表示目标中心点与图像中心点之间的偏移量。
   */
  struct Offset {
    int dx; ///< x 方向偏移量（正数表示目标在右侧，负数在左侧）
    int dy; ///< y 方向偏移量（正数表示目标在下方，负数在上方）
  };

  /**
   * @brief 计算目标框相对于图像中心的偏移量。
   *
   * @param bbox 目标边界框。
   * @param frame_width 图像宽度。
   * @param frame_height 图像高度。
   * @return Offset 偏移量结构体。
   */
  static Offset computeOffsets(const cv::Rect &bbox, int frame_width,
                               int frame_height);

  /**
   * @brief 判断偏移量是否超过阈值，从而决定是否需要移动云台。
   *
   * @param offset 偏移量。
   * @param threshold 阈值，默认值为 20。
   * @return true 偏移量超过阈值，需要移动。
   * @return false 偏移量小于等于阈值，不需要移动。
   */
  static bool needMove(const Offset &offset, int threshold = 20);
};

#endif // FACE_TRACK_GIMBAL_FACE_TRACK_CTRL_GIMBAL_CONTROLLER_H_
