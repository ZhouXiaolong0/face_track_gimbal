#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=invalid-name, no-member

"""
yolo5_inference_demo.py

YOLOv5 实时推理演示（Face Tracking 原型）

功能：
- 使用 YOLOv5 模型实时检测人脸（或其他目标）
- 在画面中心绘制十字线
- 在检测到的人脸上绘制矩形框、中心点、箭头，指示偏移量
- 在画面上显示人脸相对于中心的 dx/dy 偏移量
- 可用于测试摄像头推理性能或原型展示

模块函数：
- draw_crosshair(frame) -> (int, int)
    在图像中心绘制十字线，返回中心坐标 (cx, cy)
- draw_face(frame, box, cx, cy)
    在图像上绘制人脸检测框、中心点、偏移箭头和偏移量文字
- main()
    主函数，打开摄像头进行实时推理并可视化检测结果
"""

import cv2
from ultralytics import YOLO


def draw_crosshair(frame):
    """
    在图像中心绘制十字线，用于标记画面中心。

    参数
    ----
    frame : np.ndarray
        输入图像（BGR格式）。

    返回
    ----
    (int, int)
        图像中心坐标 (cx, cy)。
    """
    cx, cy = frame.shape[1] // 2, frame.shape[0] // 2
    cv2.line(frame, (cx - 20, cy), (cx + 20, cy), (0, 255, 0), 2)
    cv2.line(frame, (cx, cy - 20), (cx, cy + 20), (0, 255, 0), 2)
    return cx, cy


def draw_face(frame, box, cx, cy):
    """
    在图像上绘制检测到的人脸框、中心点、箭头和偏移量文字。

    参数
    ----
    frame : np.ndarray
        输入图像（BGR格式），函数会直接在图像上绘制。
    box : list/tuple/np.ndarray
        单个人脸的检测框坐标 (x_center, y_center, w, h)。
    cx : int
        图像中心 x 坐标。
    cy : int
        图像中心 y 坐标。

    返回
    ----
    None
        直接在输入图像上绘制，不返回新图像。
    """
    # 取第一个检测目标（人）
    x, y, w, h = box

    # 偏移量
    dx = int(x - cx)
    dy = int(y - cy)

    # 人脸中心点
    face_center = (int(x), int(y))

    # 计算检测框左上角和右下角坐标
    x1 = int(x - w / 2)
    y1 = int(y - h / 2)
    x2 = int(x + w / 2)
    y2 = int(y + h / 2)

    # 画检测框（绿色矩形）
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # 在画面上画人脸中心点
    cv2.circle(frame, face_center, 5, (0, 0, 255), -1)

    # 在画面上画偏移箭头（从图像中心指向人脸中心）
    cv2.arrowedLine(frame, (cx, cy), face_center, (255, 0, 0), 2)

    # 打印偏移量
    cv2.putText(
        frame,
        f"dx={dx}, dy={dy}",
        (30, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2,
    )


def main():
    """
    主函数：运行 YOLOv5 实时推理并可视化检测结果。

    功能步骤：
    1. 加载 YOLOv5 模型（指定 .pt 权重文件）。
    2. 打开摄像头，循环读取视频帧。
    3. 对每帧进行 YOLOv5 推理，检测人脸（或指定目标）。
    4. 在图像中心绘制十字线。
    5. 对每个检测到的人脸绘制：
       - 检测框
       - 人脸中心点
       - 箭头指示偏移量
       - dx/dy 偏移量文字
    6. 显示实时画面，按 'q' 键退出。

    注意事项：
    - YOLOv5 模型权重需已存在于指定路径。
    - 推理过程中会直接修改输入图像。
    - 支持摄像头 BGR 图像输入。
    """

    # 加载 YOLOv5s 模型
    model = YOLO("./models/yolov5su.pt")

    # 打开摄像头
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 直接传 BGR 图像，ultralytics 会自动处理
        results = model(frame, stream=True)

        # 在图像中心画十字线
        cx, cy = draw_crosshair(frame)

        for r in results:
            boxes = r.boxes.xywh.cpu().numpy()  # (x_center, y_center, w, h)

            if len(boxes) > 0:
                draw_face(frame, boxes[0], cx, cy)

        # 显示画面
        cv2.imshow("YOLOv5 Face Tracking Prototype", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
