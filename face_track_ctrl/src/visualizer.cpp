#include "visualizer.h"

Visualizer::Visualizer(int frame_width, int frame_height) {
  cx_ = frame_width / 2;
  cy_ = frame_height / 2;
}

void Visualizer::drawCenterCross(cv::Mat &frame, int size, cv::Scalar color) {
  cv::line(frame, cv::Point(cx_ - size, cy_), cv::Point(cx_ + size, cy_), color,
           2);
  cv::line(frame, cv::Point(cx_, cy_ - size), cv::Point(cx_, cy_ + size), color,
           2);
}

void Visualizer::drawDetections(cv::Mat &frame,
                                const std::vector<cv::Rect> &rects,
                                const std::vector<float> &scores,
                                const std::vector<int> &indices) {
  for (int idx : indices) {
    cv::Rect r = rects[idx];
    float sc = scores[idx];

    cv::rectangle(frame, r, cv::Scalar(0, 255, 0), 2);
    cv::putText(frame, cv::format("%.2f", sc), cv::Point(r.x, r.y - 5),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
  }
}

void Visualizer::drawTrackingInfo(cv::Mat &frame, const cv::Rect &target) {
  // 计算 bbox 中心
  int fx = target.x + target.width / 2;
  int fy = target.y + target.height / 2;

  // 图像中心点
  int dx = fx - cx_;
  int dy = fy - cy_;

  cv::circle(frame, cv::Point(fx, fy), 5, cv::Scalar(0, 0, 255), -1);

  // 箭头
  cv::arrowedLine(frame, cv::Point(cx_, cy_), cv::Point(fx, fy),
                  cv::Scalar(255, 0, 0), 2);

  // 偏移量文本
  cv::putText(frame, cv::format("dx=%d, dy=%d", dx, dy), cv::Point(30, 30),
              cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 255), 2);
}

void Visualizer::drawFPS(cv::Mat &frame, double fps) {
  cv::putText(frame, cv::format("FPS: %.2f", fps), cv::Point(30, 60),
              cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
}

void Visualizer::renderFrame(cv::Mat &frame, const std::vector<cv::Rect> &rects,
                             const std::vector<float> &confs,
                             const std::vector<int> &indices, double fps) {
  // 图像中心点
  int cx = frame.cols / 2;
  int cy = frame.rows / 2;

  // 绘制中心十字
  this->drawCenterCross(frame);

  // 绘制检测框和置信度
  this->drawDetections(frame, rects, confs, indices);

  // 绘制箭头和偏移量
  for (int idx : indices) {
    this->drawTrackingInfo(frame, rects[idx]);
  }

  // 绘制 FPS
  this->drawFPS(frame, fps);
}