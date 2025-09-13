#include "frame_processor.h"
#include <algorithm>

cv::Mat FrameProcessor::preprocess(const cv::Mat &src, const Options &opt,
                                   TransformMeta *meta_out) {
  CV_Assert(!src.empty());
  CV_Assert(src.channels() == 3);

  TransformMeta meta;
  meta.src_width = src.cols;
  meta.src_height = src.rows;
  meta.dst_width = opt.target_width;
  meta.dst_height = opt.target_height;

  // 1) 颜色转换（可选）
  cv::Mat color_img;
  if (opt.color_conversion >= 0) {
    cv::cvtColor(src, color_img, opt.color_conversion);
  } else {
    color_img = src;
  }

  // 2) resize or letterbox
  cv::Mat resized_or_padded;

  if (!opt.letterbox) {
    // 直接拉伸到目标尺寸
    cv::resize(color_img, resized_or_padded,
               cv::Size(opt.target_width, opt.target_height));
    // 从模型输入 -> 原图 的缩放系数
    meta.scale_x = static_cast<float>(meta.src_width) /
                   static_cast<float>(opt.target_width);
    meta.scale_y = static_cast<float>(meta.src_height) /
                   static_cast<float>(opt.target_height);
    meta.pad_left = meta.pad_top = 0;
    meta.used_letterbox = false;
  } else {
    // 按比例缩放 + 常量填充（letterbox）
    const float rW = static_cast<float>(opt.target_width) /
                     static_cast<float>(meta.src_width);
    const float rH = static_cast<float>(opt.target_height) /
                     static_cast<float>(meta.src_height);
    const float r = std::min(rW, rH);
    const int newW = static_cast<int>(std::round(meta.src_width * r));
    const int newH = static_cast<int>(std::round(meta.src_height * r));

    cv::Mat resized;
    cv::resize(color_img, resized, cv::Size(newW, newH));

    const int padX = opt.target_width - newW;
    const int padY = opt.target_height - newH;
    const int pad_left = padX / 2;
    const int pad_right = padX - pad_left;
    const int pad_top = padY / 2;
    const int pad_bottom = padY - pad_top;

    cv::copyMakeBorder(resized, resized_or_padded, pad_top, pad_bottom,
                       pad_left, pad_right, cv::BORDER_CONSTANT,
                       cv::Scalar(opt.pad_value, opt.pad_value, opt.pad_value));

    // letterbox 下，从模型输入 -> 原图 的系数（注意需要减去 padding 再除以 r）
    meta.scale_x = 1.f / r;
    meta.scale_y = 1.f / r;
    meta.pad_left = pad_left;
    meta.pad_top = pad_top;
    meta.used_letterbox = true;
  }

  // 3) 转 float / 归一化
  cv::Mat floatImg;
  resized_or_padded.convertTo(floatImg, CV_32F,
                              opt.normalize01 ? (1.0 / 255.0) : 1.0);

  if (meta_out)
    *meta_out = meta;
  return floatImg; // H x W x C，CV_32FC3
}

std::vector<float> FrameProcessor::toCHW(const cv::Mat &img) {
  CV_Assert(!img.empty());
  CV_Assert(img.type() == CV_32FC3 || img.type() == CV_32F);

  const int H = img.rows;
  const int W = img.cols;

  std::vector<float> out;
  out.reserve(3 * H * W);

  // 逐通道拷贝 (C, H, W)
  for (int c = 0; c < 3; ++c) {
    for (int y = 0; y < H; ++y) {
      const cv::Vec3f *row_ptr = img.ptr<cv::Vec3f>(y);
      for (int x = 0; x < W; ++x) {
        out.push_back(row_ptr[x][c]);
      }
    }
  }
  return out;
}

void FrameProcessor::xywh2xyxy(std::vector<cv::Vec4f> &boxes) {
  // 遍历所有边界框
  for (auto &box : boxes) {
    float cx = box[0]; // 中心点 x
    float cy = box[1]; // 中心点 y
    float w = box[2];  // 宽度
    float h = box[3];  // 高度

    // 转换为左上角 (x1,y1) + 右下角 (x2,y2) 的坐标
    box[0] = cx - w / 2.f;
    box[1] = cy - h / 2.f;
    box[2] = cx + w / 2.f;
    box[3] = cy + h / 2.f;
  }
}

cv::Rect2f FrameProcessor::mapBoxToOriginal(const cv::Rect2f &box_onInput,
                                            const TransformMeta &meta) {
  // box_onInput: x,y,w,h (在模型输入像素坐标)
  float x1 = box_onInput.x;
  float y1 = box_onInput.y;
  float x2 = box_onInput.x + box_onInput.width;
  float y2 = box_onInput.y + box_onInput.height;

  if (meta.used_letterbox) {
    // 去掉 padding，再除以 r (使用 meta.scale_x = 1/r)
    x1 = (x1 - meta.pad_left) * meta.scale_x;
    x2 = (x2 - meta.pad_left) * meta.scale_x;
    y1 = (y1 - meta.pad_top) * meta.scale_y;
    y2 = (y2 - meta.pad_top) * meta.scale_y;
  } else {
    // 直接拉伸：输入->原图比例为 src/target = meta.scale_x/meta.scale_y
    x1 *= meta.scale_x;
    x2 *= meta.scale_x;
    y1 *= meta.scale_y;
    y2 *= meta.scale_y;
  }

  // 裁切到原图范围
  x1 = std::max(0.f, std::min(x1, static_cast<float>(meta.src_width - 1)));
  x2 = std::max(0.f, std::min(x2, static_cast<float>(meta.src_width - 1)));
  y1 = std::max(0.f, std::min(y1, static_cast<float>(meta.src_height - 1)));
  y2 = std::max(0.f, std::min(y2, static_cast<float>(meta.src_height - 1)));

  return cv::Rect2f(cv::Point2f(x1, y1), cv::Point2f(x2, y2));
}

std::vector<cv::Vec4f>
FrameProcessor::mapBoxesToOriginal(const std::vector<cv::Vec4f> &boxes,
                                   const TransformMeta &meta) {

  std::vector<cv::Vec4f> boxes_mapped;
  boxes_mapped.reserve(boxes.size());

  for (const auto &b : boxes) {
    float x1 = b[0];
    float y1 = b[1];
    float x2 = b[2];
    float y2 = b[3];

    // 转成 cv::Rect2f (x,y,w,h)
    cv::Rect2f box_onInput(x1, y1, x2 - x1, y2 - y1);

    // 映射回原图
    cv::Rect2f boxOnOriginal = mapBoxToOriginal(box_onInput, meta);

    // 保存回 xyxy 格式
    boxes_mapped.push_back(cv::Vec4f(boxOnOriginal.x, boxOnOriginal.y,
                                     boxOnOriginal.x + boxOnOriginal.width,
                                     boxOnOriginal.y + boxOnOriginal.height));
  }

  return boxes_mapped;
}

FrameProcessor::NMSResult
FrameProcessor::nms(const std::vector<cv::Vec4f> &boxes,
                    const std::vector<float> &scores, float conf_threshold,
                    float iou_threshold) {
  NMSResult result;

  // 转换 boxes -> cv::Rect
  for (size_t i = 0; i < boxes.size(); ++i) {
    result.rects.emplace_back(
        cv::Point(static_cast<int>(boxes[i][0]), static_cast<int>(boxes[i][1])),
        cv::Point(static_cast<int>(boxes[i][2]),
                  static_cast<int>(boxes[i][3])));
    result.confs.push_back(scores[i]);
  }

  // NMS
  cv::dnn::NMSBoxes(result.rects, result.confs, conf_threshold, iou_threshold,
                    result.indices);

  return result;
}

FrameProcessor::Processed FrameProcessor::processFrame(const cv::Mat &frame) {
  Processed result;

  // 调用已有的 preprocess，拿到 processed_frame 和 meta
  result.processed_frame = FrameProcessor::preprocess(frame, &result.meta);

  return result;
}