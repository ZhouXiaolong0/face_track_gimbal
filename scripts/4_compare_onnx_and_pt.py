#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=no-member

"""
compare_onnx_and_pt.py

该模块用于比较 YOLOv5 模型的 PyTorch (FP16) 与 ONNX 版本在推理结果上的差异。

功能
----
1. 加载并验证 YOLOv5 的 ONNX 模型。
2. 初始化 PyTorch YOLOv5 (FP16) 和 ONNX YOLOv5 模型。
3. 从本地摄像头捕获一帧图像。
4. 分别使用 PyTorch 和 ONNX 模型进行推理，得到预测框。
5. 对比两者的推理结果，包括：
   - 预测框数量
   - 坐标数值差异
   - 平均 IoU（Intersection over Union）

依赖
----
- onnx
- opencv-python (cv2)
- numpy
- ultralytics (YOLOv5)

输入
----
- YOLOv5 PyTorch 模型文件 (.pt)
- YOLOv5 ONNX 模型文件 (.onnx)
- 摄像头实时捕获的图像

输出
----
- 控制台打印对比结果，包括预测框数量差异、最大坐标偏差和平均 IoU。

适用场景
--------
该模块适用于验证 YOLOv5 模型在 PyTorch 与 ONNX 部署环境下的一致性，
帮助开发者确认模型转换是否准确，避免精度损失。
"""


import onnx
import cv2
import numpy as np
from ultralytics import YOLO


def iou(box1, box2):
    """
    计算两个矩形框的 IoU（Intersection over Union）。

    定义
    ----
    IoU = (交集面积) / (并集面积)

    参数
    ----
    box1 : list[float] 或 tuple[float]
        第一个矩形框，格式为 [x1, y1, x2, y2]，
        其中 (x1, y1) 为左上角坐标，(x2, y2) 为右下角坐标。
    box2 : list[float] 或 tuple[float]
        第二个矩形框，格式同上。

    返回
    ----
    float
        IoU 值，范围 [0, 1]。
        - 0 表示没有重叠
        - 1 表示两个框完全相同

    注意
    ----
    - 坐标需满足 x1 < x2, y1 < y2，否则计算结果可能不正确。
    - 当两个框无交集时，返回 0。
    """
    x_a = max(box1[0], box2[0])
    y_a = max(box1[1], box2[1])
    x_b = min(box1[2], box2[2])
    y_b = min(box1[3], box2[3])

    inter = max(0, x_b - x_a) * max(0, y_b - y_a)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / union if union != 0 else 0


def main():
    """
    主函数：比较 PyTorch YOLOv5su 和 ONNX YOLOv5su 模型的推理结果

    步骤:
        1. 验证并加载 ONNX 模型。
        2. 初始化 PyTorch 和 ONNX 格式的 YOLOv5 模型。
        3. 从摄像头采集一帧图像。
        4. 使用 PyTorch 模型进行推理并获取预测框。
        5. 使用 ONNX 模型进行推理并获取预测框。
        6. 对比 PyTorch 与 ONNX 模型的检测结果，包括：
           - 预测框数量
           - 坐标数值差异
           - 预测框 IoU（交并比）
    """
    # 1. 验证模型
    onnx_model_path = "./models/yolov5su.onnx"
    onnx_model = onnx.load(onnx_model_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model is valid!")

    # =========================
    # 1️⃣ 初始化模型
    # =========================
    # PyTorch YOLOv5su
    pt_model = YOLO("./models/yolov5su.pt")
    pt_model.model.half()

    # ONNX YOLOv5su
    onnx_model = YOLO(onnx_model_path)

    # =========================
    # 2️⃣ 从摄像头获取一帧
    # =========================
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("无法读取摄像头图片！")

    # =========================
    # 3️⃣ PyTorch 推理
    # =========================
    pt_results = pt_model.predict(frame)
    pt_boxes = pt_results[0].boxes.xyxy.numpy()
    print("PyTorch预测框:", pt_boxes)

    # =========================
    # 4️⃣ ONNX 推理
    # =========================
    onnx_results = onnx_model.predict(frame)
    onnx_boxes = onnx_results[0].boxes.xyxy.numpy()
    print("ONNX预测框:", onnx_boxes)

    # =========================
    # 5️⃣ 简单对比
    # =========================
    print("PyTorch框数:", pt_boxes.shape[0])
    print("ONNX框数:", onnx_boxes.shape[0])

    # 可选：数值差异
    if pt_boxes.shape[0] == onnx_boxes.shape[0]:
        diff = abs(pt_boxes - onnx_boxes)
        print("预测框数值差异 max:", diff.max())

    if pt_boxes.shape == onnx_boxes.shape:
        # 坐标差异
        diff = np.abs(pt_boxes - onnx_boxes)
        print("坐标最大差:", diff.max())

        # IoU
        ious = [iou(p, o) for p, o in zip(pt_boxes, onnx_boxes)]
        print("平均IoU:", np.mean(ious))


if __name__ == "__main__":
    main()
