#ifndef FACE_TRACK_GIMBAL_FACE_TRACK_CTRL_FRAME_PROCESSOR_H_
#define FACE_TRACK_GIMBAL_FACE_TRACK_CTRL_FRAME_PROCESSOR_H_

#include <opencv2/opencv.hpp>
#include <vector>

/**
 * @class FrameProcessor
 * @brief 视觉前处理工具：颜色空间转换、resize/letterbox、归一化、HWC->CHW。
 */
class FrameProcessor {
public:
  /**
   * @brief Non-Maximum Suppression (NMS) 结果结构体
   *
   * 用于保存经过 NMS 后的检测结果，包括保留下来的索引、
   * 对应的矩形框以及置信度。
   *
   */
  struct NMSResult {
    std::vector<int> indices;    ///< NMS 后保留下来的索引
    std::vector<cv::Rect> rects; ///< 对应的矩形框
    std::vector<float> confs;    ///< 对应的置信度
  };

  /**
   * @brief 前处理可选参数
   *
   */
  struct Options {
    int target_width;        ///< 模型输入所需要的宽度
    int target_height;       ///< 模型输入所需要的高度
    int color_conversion;    ///< 颜色空间转换，-1 表示不转换
    bool normalize01;        ///< 是否缩放到 [0,1]
    bool letterbox;          ///< 是否使用 letterbox（保比例填充）
    unsigned char pad_value; ///< letterbox 填充值（YOLO 常用 114）

    // 显式构造函数（避免某些编译器/标准问题）
    Options()
        : target_width(640), target_height(640),
          color_conversion(cv::COLOR_BGR2RGB), normalize01(true),
          letterbox(false), pad_value(114) {}
  };

  /**
   * @brief 变换元数据（用于把检测框从模型输入坐标映回原图）
   */
  struct TransformMeta {
    int src_width;       ///< 原图尺寸宽
    int src_height;      ///< 原图尺寸高
    int dst_width;       ///< 前处理后的尺寸（一般等于 target）宽
    int dst_height;      ///< 前处理后的尺寸（一般等于 target）高
    float scale_x;       ///< 从模型输入 -> 原图 的缩放系数
    float scale_y;       ///< 从模型输入 -> 原图 的缩放系数
    int pad_left;        ///< letterbox 左填充
    int pad_top;         ///< letterbox 上填充
    bool used_letterbox; ///< 是否使用了 letterbox

    TransformMeta()
        : src_width(0), src_height(0), dst_width(0), dst_height(0),
          scale_x(1.f), scale_y(1.f), pad_left(0), pad_top(0),
          used_letterbox(false) {}
  };

  /**
   * @brief 封装处理后的帧及其相关的元数据信息。
   *
   */
  struct Processed {
    cv::Mat processed_frame;
    TransformMeta meta;
  };

  /**
   * @brief 前处理：颜色转换 -> resize/letterbox -> 归一化 (到 float32)
   * @param src   输入原图（BGR 常见）
   * @param opt   前处理选项
   * @param meta_out 若不为 nullptr，则输出用于回映射的元数据
   * @return 处理后的图像（CV_32F，H×W×C，若 normalize01=true 范围为[0,1]）
   */
  static cv::Mat preprocess(const cv::Mat &src, const Options &opt = Options(),
                            TransformMeta *meta_out = nullptr);

  /**
   * @brief 重载：使用默认 Options
   */
  static cv::Mat preprocess(const cv::Mat &src,
                            TransformMeta *meta_out = nullptr) {
    Options def;
    return preprocess(src, def, meta_out);
  }

  /**
   * @brief 将 HWC（三通道 float 图）打平成 CHW 的连续向量
   * @param img  预处理后的图（CV_32F，3通道）
   * @return 按 CHW 顺序排列的一维数组（size = 3 * H * W）
   */
  static std::vector<float> toCHW(const cv::Mat &img);

  /**
   * @brief 将一组边界框从 [中心点 x, 中心点 y, 宽, 高] 转换为 [x1, y1, x2, y2]
   *
   * 这个函数会直接修改输入的 boxes 向量，每个元素的坐标顺序会被替换。
   * 适用于 YOLO 模型输出的标准格式转换。
   *
   * @param boxes 输入/输出边界框向量，每个 cv::Vec4f 表示一个框：
   * 原始格式: [cx, cy, w, h]，转换后格式: [x1, y1, x2, y2]
   */
  static void xywh2xyxy(std::vector<cv::Vec4f> &boxes);

  /**
   * @brief 将模型输入坐标系中的框（xywh/xyxy 均可）映射回原图坐标
   * @param box_onInput  模型输入坐标系中的框（使用 Rect2f 即可）
   * @param meta        预处理时的元数据
   * @return 映射回原图的框（xyxy）
   */
  static cv::Rect2f mapBoxToOriginal(const cv::Rect2f &box_onInput,
                                     const TransformMeta &meta);

  /**
   * @brief 将一组边界框从模型输入坐标映射回原图坐标
   *
   * 输入输出格式均为 xyxy: [x1, y1, x2, y2]
   * 内部会调用 mapBoxToOriginal 处理 padding 和缩放。
   *
   * @param boxes 输入边界框向量（xyxy）
   * @param meta 预处理时记录的 TransformMeta
   * @return std::vector<cv::Vec4f> 映射回原图后的边界框（xyxy）
   */
  static std::vector<cv::Vec4f>
  mapBoxesToOriginal(const std::vector<cv::Vec4f> &boxes,
                     const TransformMeta &meta);

  /**
   * @brief 对检测框进行非极大值抑制 (NMS)
   *
   * 将输入的 boxes 和 scores 进行筛选，去掉重叠度过高的框，
   * 仅保留置信度高且不重叠的目标框。
   *
   * @param boxes 输入的检测框，格式为 [x1, y1, x2, y2]
   * @param scores 每个检测框对应的置信度
   * @param conf_threshold 置信度阈值，小于该值的框会被丢弃
   * @param iou_threshold IOU 阈值，重叠度高于该值的框会被抑制
   * @return FrameProcessor::NMSResult 返回经过 NMS
   * 筛选后的结果，包括保留索引、矩形框和置信度
   */
  static FrameProcessor::NMSResult nms(const std::vector<cv::Vec4f> &boxes,
                                       const std::vector<float> &scores,
                                       float conf_threshold,
                                       float iou_threshold);

  static Processed processFrame(const cv::Mat &frame);
};

#endif // FACE_TRACK_GIMBAL_FACE_TRACK_CTRL_FRAME_PROCESSOR_H_