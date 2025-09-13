#ifndef FACE_TRACK_GIMBAL_FACE_TRACK_CTRL_VISUALIZER_H_
#define FACE_TRACK_GIMBAL_FACE_TRACK_CTRL_VISUALIZER_H_

#include <opencv2/opencv.hpp>
#include <vector>

/**
 * @brief Visualizer 类，用于在图像上绘制检测结果、中心十字、FPS 和跟踪信息
 *
 */
class Visualizer {
public:
  /**
   * @brief 构造函数
   *
   * @param frame_width 图像宽度
   * @param frame_height 图像高度
   */
  Visualizer(int frame_width, int frame_height);

  /**
   * @brief 绘制图像中心十字线
   *
   * @param frame 要绘制的图像
   * @param size 十字线长度，默认 20
   * @param color 颜色，默认绿色 (0, 255, 0)
   */
  void drawCenterCross(cv::Mat &frame, int size = 20,
                       cv::Scalar color = cv::Scalar(0, 255, 0));

  /**
   * @brief 绘制检测框和置信度
   *
   * @param frame 要绘制的图像
   * @param rects 检测框集合
   * @param scores 对应每个检测框的置信度
   * @param indices 要绘制的检测框索引
   */
  void drawDetections(cv::Mat &frame, const std::vector<cv::Rect> &rects,
                      const std::vector<float> &scores,
                      const std::vector<int> &indices);

  /**
   * @brief 绘制目标中心点、偏移箭头和偏移值
   *
   * @param frame 要绘制的图像
   * @param target 目标检测框
   */
  void drawTrackingInfo(cv::Mat &frame, const cv::Rect &target);

  /**
   * @brief 绘制 FPS
   *
   * @param frame 要绘制的图像
   * @param fps 实时帧率
   */
  void drawFPS(cv::Mat &frame, double fps);

  /**
   * @brief 渲染整帧图像，包括检测框、置信度、中心十字和 FPS
   *
   * @param frame 要渲染的图像
   * @param rects 检测框集合
   * @param confs 对应每个检测框的置信度
   * @param indices 要渲染的检测框索引
   * @param fps 实时帧率
   */
  void renderFrame(cv::Mat &frame, const std::vector<cv::Rect> &rects,
                   const std::vector<float> &confs,
                   const std::vector<int> &indices, double fps);

private:
  int cx_; /**< 图像中心点 x */
  int cy_; /**< 图像中心点 y */
};

#endif // FACE_TRACK_GIMBAL_FACE_TRACK_CTRL_VISUALIZER_H_