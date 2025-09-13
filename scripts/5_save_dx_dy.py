#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=no-member,

"""
onnx_yolov5_tracking.py

基于 ONNX YOLOv5 模型的实时人体检测与偏移量记录模块。

功能
----
1. 从摄像头实时采集视频帧。
2. 使用 YOLOv5su ONNX 模型进行人体检测。
3. 在图像上绘制检测框、中心点和偏移箭头。
4. 计算检测目标中心点相对于画面中心的偏移量 (dx, dy)。
5. 将偏移量随帧序号记录到 CSV 文件。
6. 在窗口中实时显示检测结果。

依赖
----
- Python 3.7+
- OpenCV (cv2)
- NumPy
- onnxruntime
- csv（标准库）

输入
----
- 摄像头视频流（默认索引 0）

输出
----
- 实时窗口显示检测结果
- CSV 日志文件: `./log/dx_dy_log.csv`
  * frame: 帧序号
  * dx: x 方向偏移量
  * dy: y 方向偏移量

用法
----
直接运行该脚本：

    $ python onnx_yolov5_tracking.py

运行后会弹出窗口显示检测结果，按 'q' 键退出。

异常
----
- RuntimeError: 如果无法打开摄像头
"""

import csv
import cv2
import numpy as np
import onnxruntime as ort


def nms(boxes, scores, iou_threshold=0.45):  # pylint: disable=too-many-locals
    """
    执行非极大值抑制（Non-Maximum Suppression, NMS）以去除重叠过多的检测框。

    参数:
        boxes (np.ndarray): 检测框坐标，形状 (N,4)，格式为 xyxy
        scores (np.ndarray): 每个检测框的置信度，形状 (N,)
        iou_threshold (float, optional): IOU 阈值，超过该阈值的框会被抑制，默认 0.45

    返回:
        list: 保留下来的检测框索引列表

    功能:
        - 按置信度从高到低排序
        - 遍历每个框，移除与当前框重叠度超过 iou_threshold 的其他框
        - 返回最终保留的框的索引
    """
    # x1 = boxes[:, 0]
    # y1 = boxes[:, 1]
    # x2 = boxes[:, 2]
    # y2 = boxes[:, 3]
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]  # 按分数降序排序

    keep = []
    while order.size > 0:
        i = order[0]  # 当前最大分数索引
        keep.append(i)  # 保留
        # 计算与剩余框的交集
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        # IOU = 交集 / 并集
        iou = inter / (areas[i] + areas[order[1:]] - inter)

        # 只保留 IOU 小于阈值的索引
        order = order[np.where(iou <= iou_threshold)[0] + 1]
    return keep


def preprocess_frame(frame, input_size=640):
    """
    对输入图像进行 YOLO 模型所需的预处理。

    参数:
        frame (np.ndarray): 原始 BGR 图像
        input_size (int, optional): 模型输入尺寸，默认 640，表示将图像缩放为 input_size x input_size

    返回:
        np.ndarray: 预处理后的图像，形状为 (1, 3, input_size, input_size)，
                    数据类型 float32，像素值归一化到 [0,1]

    功能:
        - 将 BGR 图像转换为 RGB
        - 缩放到模型输入尺寸
        - HWC -> CHW 转换
        - 增加 batch 维度
        - 像素值归一化到 [0,1]
    """
    # BGR -> RGB
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # 缩放到 YOLO 输入尺寸 (640x640)
    img_resized = cv2.resize(img, (input_size, input_size))
    # HWC -> CHW, 增加 batch 维度, 归一化到 [0,1]
    img_input = img_resized.transpose(2, 0, 1)[None, ...].astype(np.float32) / 255.0
    return img_input


# --- Inference ---
def run_inference(session, input_name, img_input):
    """
    使用 ONNX 模型对输入图像进行推理，返回检测框和对应置信度。

    参数:
        session (onnxruntime.InferenceSession): 已加载的 ONNX 模型会话
        input_name (str): 模型输入节点名称
        img_input (np.ndarray): 预处理后的图像，形状 (1, 3, H, W)，dtype=float32

    返回:
        tuple:
            - boxes (np.ndarray): 检测框，格式为 xywh，形状 (N,4)
            - scores (np.ndarray): 对应每个检测框的置信度，形状 (N,)

    功能:
        - 使用 ONNX Runtime 对输入图像进行前向推理
        - 解码输出为 bbox 和类别 logits
        - 计算每个 bbox 对应的人类别置信度
        - 返回 bbox 和 scores
    """
    # 使用 ONNX Runtime 推理
    outputs = session.run(None, {input_name: img_input})[0]  # 输出 shape: (1,84,8400)
    pred = outputs.squeeze(0).T  # 转成 (8400,84)，每行是一个 anchor 的预测

    # print("pred is", pred.shape)

    # --- Decode --- (官方 ONNX 输出后处理)
    boxes = pred[:, 0:4]  # xywh
    objectness = pred[:, 4]  # 目标置信度
    class_logits = pred[:, 5:]  # 分类概率

    # --- 只保留人类别 ---
    human_logits = class_logits[:, 0]
    scores = objectness + human_logits  # class_probs[:,0] = 人类别概率

    return boxes, scores


def filter_boxes(boxes, scores, image_size, conf_threshold=0.3, iou_threshold=0.45):
    """
    对 YOLO 模型输出的检测框进行置信度筛选、坐标转换和非极大值抑制 (NMS)。

    参数:
        boxes (np.ndarray): 模型输出的 bbox，格式为 xywh (中心点坐标 + 宽高)，形状 (N,4)
        scores (np.ndarray): 每个 bbox 的置信度，形状 (N,)
        image_size (tuple): 原图尺寸 (height, width)
        conf_thres (float, optional): 置信度阈值，低于该值的框会被丢弃，默认 0.3
        iou_thres (float, optional): NMS 的 IOU 阈值，默认 0.45

    返回:
        tuple:
            - boxes (np.ndarray): 经过筛选和 NMS 后的 bbox，格式 xyxy，形状 (M,4)
            - scores (np.ndarray): 对应的置信度，形状 (M,)

    功能:
        1. 根据 conf_thres 筛选低置信度框
        2. 将 bbox 坐标从 xywh 转换为 xyxy
        3. 将坐标映射回原图尺寸
        4. 使用 NMS 删除重叠过多的框
    """
    h0, w0 = image_size
    mask = scores > conf_threshold
    # boxes = boxes[mask]
    # scores = scores[mask]
    boxes, scores = boxes[mask], scores[mask]

    if len(boxes) == 0:
        return np.array([]), np.array([])

    # xywh -> xyxy
    boxes[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    boxes[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
    boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

    # 映射回原图
    boxes[:, [0, 2]] *= w0 / 640
    boxes[:, [1, 3]] *= h0 / 640

    # NMS
    keep = nms(boxes, scores, iou_threshold)
    return boxes[keep], scores[keep]


def draw_center(frame):
    """
    在图像中心绘制十字线标记。

    参数:
        frame (np.ndarray): 当前帧图像

    返回:
        tuple: 图像中心点坐标 (cx, cy)

    功能:
        - 计算图像中心点
        - 在中心绘制绿色十字线
    """
    h0, w0 = frame.shape[:2]
    # 图像中心点
    cx, cy = w0 // 2, h0 // 2
    # 画十字线标记中心
    cv2.line(frame, (cx - 20, cy), (cx + 20, cy), (0, 255, 0), 2)
    cv2.line(frame, (cx, cy - 20), (cx, cy + 20), (0, 255, 0), 2)
    return cx, cy


def draw_box(frame_img, box, score):
    """
    在图像上绘制检测框和置信度分数。

    参数：
        frame_img (numpy.ndarray): 待绘制的图像帧 (BGR 格式)。
        box (tuple | list): 检测框坐标 (x1, y1, x2, y2)，左上角和右下角像素点。
        score (float): 检测置信度分数，将显示在检测框上方。

    返回：
        None: 该函数会直接修改输入的图像 frame_img，而不会返回新图像。
    """
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(frame_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(
        frame_img,
        f"{score:.2f}",
        (x1, y1 - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )


def log_and_arrow(frame_img, context, result):
    """
    在图像上绘制检测中心点、偏移箭头，并记录偏移量日志。

    参数：
        frame_img (numpy.ndarray): 当前帧图像 (BGR 格式)，函数会直接在其上绘制。
        context (dict): 上下文信息，包含以下键：
            - "cx" (int): 图像中心点的 x 坐标。
            - "cy" (int): 图像中心点的 y 坐标。
            - "frame_index" (int): 当前帧的索引编号。
            - "writer" (csv.writer): 用于写入日志的 CSV writer 对象。
        result (dict): 检测结果，包含以下键：
            - "fx" (int): 检测到的人体中心点 x 坐标。
            - "fy" (int): 检测到的人体中心点 y 坐标。
            - "dx" (int): x 方向的偏移量 (fx - cx)。
            - "dy" (int): y 方向的偏移量 (fy - cy)。

    返回：
        None: 函数直接修改输入图像并写入日志，不返回值。
    """
    cx, cy = context["cx"], context["cy"]
    frame_index, writer = context["frame_index"], context["writer"]

    fx, fy = result["fx"], result["fy"]
    dx, dy = result["dx"], result["dy"]

    # 人中心点
    cv2.circle(frame_img, (fx, fy), 5, (0, 0, 255), -1)

    # 写入日志
    writer.writerow([frame_index, dx, dy])

    # 画箭头
    cv2.arrowedLine(frame_img, (cx, cy), (fx, fy), (255, 0, 0), 2)

    # 显示偏移量
    cv2.putText(
        frame_img,
        f"dx={dx}, dy={dy}",
        (30, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2,
    )


def draw_and_log(frame_img, det_boxes, det_scores, context):
    """
    绘制检测框、目标中心点、偏移量，并写入日志。

    功能:
        - 遍历检测框和置信度，逐一绘制矩形框。
        - 计算检测目标的中心点 (fx, fy)。
        - 计算目标与画面中心点 (cx, cy) 的偏移量 (dx, dy)。
        - 调用 log_and_arrow 画出箭头，并记录日志。

    参数:
        frame_img (numpy.ndarray): 当前帧图像。
        det_boxes (list[tuple[float]]): 检测框坐标列表，每个框为 (x1, y1, x2, y2)。
        det_scores (list[float]): 与检测框对应的置信度分数。
        context (dict): 上下文信息，包含:
            - cx (int): 画面中心点的 X 坐标。
            - cy (int): 画面中心点的 Y 坐标。
            - frame_index (int): 当前帧索引。
            - writer (csv.writer): 用于写入日志的 CSV writer。

    返回:
        None
    """
    cx, cy = int(context["cx"]), int(context["cy"])

    for box, score in zip(det_boxes, det_scores):
        # 绘制检测框
        draw_box(frame_img, tuple(map(int, box)), score)

        # 生成结果字典，计算中心点和偏移量
        result = {
            "fx": int((box[0] + box[2]) // 2),
            "fy": int((box[1] + box[3]) // 2),
            "dx": int((box[0] + box[2]) // 2) - cx,
            "dy": int((box[1] + box[3]) // 2) - cy,
        }

        # 调用绘制箭头和写日志函数
        log_and_arrow(frame_img, context, result)


def process_frame(frame, context):
    """
    处理单帧图像：完成推理、NMS 过滤、绘制结果并记录偏移量。

    参数
    ----------
    frame : np.ndarray
        当前帧的图像数据 (BGR 格式，来自 OpenCV)。
    context : dict
        上下文字典，包含必要的推理与绘制参数：
        - "session" : onnxruntime.InferenceSession
            ONNX 推理会话对象
        - "input_name" : str
            模型输入的名称
        - "conf_threshold" : float, 可选
            置信度阈值，默认 0.3
        - "iou_threshold" : float, 可选
            NMS 的 IoU 阈值，默认 0.45
        - 以及 draw_and_log 所需的中心点、日志 writer 等信息

    功能
    ----
    1. 对输入帧进行预处理以符合模型输入
    2. 调用 ONNX 模型进行推理，获得检测框和置信度
    3. 使用置信度阈值和 NMS 过滤检测框
    4. 绘制画面中心点
    5. 绘制检测框、中心点、偏移量，并写入日志
    """
    session = context["session"]
    input_name = context["input_name"]
    conf_threshold = context.get("conf_threshold", 0.3)
    iou_threshold = context.get("iou_threshold", 0.45)

    img_input = preprocess_frame(frame)
    boxes, scores = run_inference(session, input_name, img_input)
    boxes, scores = filter_boxes(
        boxes, scores, frame.shape[:2], conf_threshold, iou_threshold
    )

    draw_center(frame)
    draw_and_log(frame, boxes, scores, context)


def main():
    """
    主程序入口。

    功能：
        - 加载 YOLOv5su ONNX 模型，并初始化推理会话
        - 打开默认摄像头，实时采集视频帧
        - 对每一帧图像进行预处理、推理和后处理
        - 绘制检测结果与图像中心点
        - 计算目标中心点与画面中心的偏移量 (dx, dy)，并记录到 CSV 文件
        - 在窗口中实时显示检测结果

    输入：
        无（直接从默认摄像头读取图像）

    输出：
        - 屏幕实时显示检测结果窗口
        - 保存日志文件 ./log/dx_dy_log.csv，包含以下字段：
            * frame: 帧序号
            * dx: 目标中心点相对画面中心的水平偏移量
            * dy: 目标中心点相对画面中心的垂直偏移量

    退出方式：
        - 在窗口中按下 'q' 键退出程序

    异常：
        RuntimeError: 如果无法打开摄像头
    """

    # -----------------------------
    # 1️⃣ 初始化 ONNX 模型
    # -----------------------------
    onnx_path = "./models/yolov5su.onnx"
    # 创建 ONNX Runtime 推理会话，指定使用 CPU
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    # 获取模型输入名，用于后续推理
    input_name = session.get_inputs()[0].name

    # 置信度阈值
    conf_threshold = 0.3
    # NMS IOU 阈值
    iou_threshold = 0.45

    # -----------------------------
    # 3️⃣ 摄像头初始化
    # -----------------------------
    cap = cv2.VideoCapture(0)  # 打开默认摄像头
    if not cap.isOpened():
        raise RuntimeError("无法打开摄像头")

    # -----------------------------
    # 4️⃣ 循环推理
    # -----------------------------
    with open("./log/dx_dy_log.csv", "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        # 写表头
        writer.writerow(["frame", "dx", "dy"])

        frame_idx = 0
        while True:
            ret, frame = cap.read()  # 读取一帧图像
            if not ret:
                break

            context = {
                "cx": frame.shape[1] // 2,
                "cy": frame.shape[0] // 2,
                "frame_index": frame_idx,
                "writer": writer,
                "session": session,
                "input_name": input_name,
                "conf_threshold": conf_threshold,
                "iou_threshold": iou_threshold,
            }

            process_frame(frame, context)

            frame_idx += 1

            # 显示结果
            cv2.imshow("ONNX YOLOv5su", frame)
            # 按 'q' 退出
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
