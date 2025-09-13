#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=no-member, invalid-name

"""
人脸追踪二维云台模块 (Face Track Gimbal)

功能:
    - 使用 YOLOv5 ONNX 模型进行实时人脸检测
    - 通过 OpenCV 获取摄像头视频帧
    - 计算检测框中心点偏移量(dx, dy)
    - 将偏移量通过串口发送给舵机控制器
    - 在画面上绘制检测框、中心点、箭头和偏移量
    - 支持非极大值抑制 (NMS) 去除重复检测框
    - 可通过串口与 STM32 或 Jetson Nano 连接进行舵机控制

依赖:
    - Python 3.x
    - OpenCV (cv2)
    - NumPy
    - ONNX Runtime (onnxruntime)
    - pySerial (serial)

主要函数:
    - setup_camera: 初始化摄像头并返回帧尺寸
    - setup_serial: 初始化串口连接
    - preprocess_frame: 图像预处理，BGR -> RGB, 缩放, CHW -> batch
    - run_inference: 使用 ONNX 模型进行推理
    - filter_boxes: 置信度筛选和 NMS
    - draw_center: 绘制画面中心十字线
    - compute_offsets: 计算检测框中心点和偏移量
    - draw_box_and_arrow: 绘制检测框、中心点、箭头和偏移量
    - send_offsets: 将偏移量发送到串口
    - process_detections: 处理所有检测框并绘制与发送数据
    - main: 主函数，执行摄像头捕获、模型推理和可视化
"""

import sys
import time

import cv2
import numpy as np
import onnxruntime as ort
import serial


def setup_camera(camera_id=0, width=None, height=None):
    """
    初始化摄像头。

    参数:
        camera_id (int): 摄像头索引，默认 0
        width (int, optional): 设置摄像头宽度
        height (int, optional): 设置摄像头高度

    返回:
        cap (cv2.VideoCapture): 已打开的摄像头对象
        w0 (int): 摄像头实际宽度
        h0 (int): 摄像头实际高度
    """
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开摄像头 {camera_id}")

    # 可选设置分辨率
    if width is not None:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height is not None:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # 获取实际分辨率
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("摄像头读取帧失败")
    h0, w0 = frame.shape[:2]
    print(f"摄像头分辨率: width={w0}, height={h0}")

    return cap, w0, h0


def setup_serial(port, baudrate):
    """
    初始化串口连接。

    参数:
        port (str): 串口号，例如 'COM3' 或 '/dev/ttyUSB0'
        baudrate (int): 波特率，例如 115200

    返回:
        serial.Serial: 已打开的串口对象

    异常:
        如果打开串口失败，会打印错误信息并退出程序。
    """
    try:
        ser = serial.Serial(port, baudrate, timeout=1)
        print(f"串口 {port} 打开成功！")
        return ser
    except serial.SerialException as e:
        print(f"打开串口失败: {e}")
        sys.exit(1)


def send_offsets(ser, dx, dy):
    """
    发送偏移量数据到串口。

    参数:
        ser (serial.Serial): 已打开的串口对象
        dx (int): x 方向偏移量
        dy (int): y 方向偏移量

    功能:
        将 dx 和 dy 构造成字符串 "dx,dy\n" 并通过串口发送。
        发送后会打印发送的字节数和内容。
        包含 10ms 延时和每秒一次的发送间隔。
    """
    # 随机生成 dx, dy 模拟偏移量
    # dx = random.randint(-240, 240)
    # dy = random.randint(-180, 180)

    # 构造字符串，格式：dx,dy\n
    send_str = f"{dx},{dy}\n"

    # 发送
    # ser.write(send_str.encode('ascii'))
    n = ser.write(send_str.encode("ascii"))
    time.sleep(0.01)  # 10ms 延时
    print(f"发送了 {n} 个字节，已发送: {send_str.strip()}")

    time.sleep(1)  # 每秒发送一次


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
    # x1 = boxes[:,0]
    # y1 = boxes[:,1]
    # x2 = boxes[:,2]
    # y2 = boxes[:,3]
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
    # 使用 ONNX Runtime 进行前向推理
    outputs = session.run(None, {input_name: img_input})[0]  # 输出 shape: (1,84,8400)

    # 去掉 batch 维度并转置，使每行对应一个 anchor 的预测
    pred = outputs.squeeze(0).T  # 转成 (8400,84)，每行是一个 anchor 的预测

    print("pred is", pred.shape)

    # --- Decode --- (官方 ONNX 输出后处理)
    boxes = pred[:, 0:4]  # 前四个是 bbox 的 xywh
    objectness = pred[:, 4]  # 第五个是目标置信度
    class_logits = pred[:, 5:]  # 80类  # 后面是 80 个类概率

    # --- 计算每个框的最终置信度 ---
    # 这里只保留“人”类别
    # 只保留人类别
    human_logits = class_logits[:, 0]
    scores = objectness + human_logits  # class_probs[:,0] = 人类别概率

    return boxes, scores


def filter_boxes(boxes, scores, image_size, conf_thres=0.3, iou_thres=0.45):
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
    # 置信度筛选
    h0, w0 = image_size
    mask = scores > conf_thres  # 只保留置信度大于阈值的框
    # boxes = boxes[mask]
    # scores = scores[mask]
    boxes, scores = boxes[mask], scores[mask]

    if len(boxes) == 0:
        return np.array([]), np.array([])

    # --- 坐标转换 xywh -> xyxy ---
    boxes[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x_center -> x1
    boxes[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y_center -> y1
    boxes[:, 2] = boxes[:, 0] + boxes[:, 2]  # x2
    boxes[:, 3] = boxes[:, 1] + boxes[:, 3]  # y2

    # --- 映射回原图尺寸 ---
    boxes[:, [0, 2]] *= w0 / 640
    boxes[:, [1, 3]] *= h0 / 640

    print(len(boxes))

    # --- NMS 非极大值抑制 ---
    # if len(boxes) > 0:
    keep = nms(boxes, scores, iou_threshold=iou_thres)  # 返回保留的索引
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


def compute_offsets(box, frame_shape):
    """
    计算检测框的中心点坐标及相对于图像中心的偏移量。

    参数:
        box (iterable): 检测框坐标，格式为 xyxy
        frame_shape (tuple or np.ndarray): 当前帧的形状 (height, width, channels)

    返回:
        tuple:
            - center (tuple): 检测框中心点坐标 (fx, fy)
            - offset (tuple): 相对于图像中心的偏移量 (dx, dy)

    功能:
        - 计算 bbox 的中心点坐标
        - 绘制中心点到图像上 (红色圆点)
        - 计算中心点相对于图像中心的偏移量 dx 和 dy
    """
    x1, y1, x2, y2 = map(int, box)
    # 计算 bbox 中心点
    fx, fy = (x1 + x2) // 2, (y1 + y2) // 2
    h0, w0 = frame_shape[:2]
    cv2.circle(frame_shape, (fx, fy), 5, (0, 0, 255), -1)

    # 计算偏移量
    # dx 最小值 -320, 最大值 320
    # dy 最小值 -240，最大值 240
    cx, cy = w0 // 2, h0 // 2
    dx, dy = fx - cx, fy - cy
    return (fx, fy), (dx, dy)


def draw_box_and_arrow(frame, box, score, center, offset):
    """
    在图像上绘制检测框、置信度、中心点、箭头以及偏移量。

    参数:
        frame (np.ndarray): 当前帧图像
        box (iterable): 检测框，格式为 xyxy
        score (float): 检测框置信度
        center (tuple): 检测框中心点坐标 (fx, fy)
        offset (tuple): 偏移量 (dx, dy)，相对于图像中心点

    功能:
        - 绘制绿色检测框和置信度
        - 绘制红色中心点
        - 绘制从图像中心到检测框中心的蓝色箭头
        - 在左上角显示偏移量 dx 和 dy
    """
    x1, y1, x2, y2 = map(int, box)
    dx, dy = offset

    # 绘制检测框
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    # 显示置信度
    cv2.putText(
        frame,
        f"{score:.2f}",
        (x1, y1 - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )
    cv2.circle(frame, center, 5, (0, 0, 255), -1)
    # 箭头
    h0, w0 = frame.shape[:2]
    cx, cy = w0 // 2, h0 // 2
    # 画箭头：从画面中心 -> 人脸中心
    cv2.arrowedLine(frame, (cx, cy), center, (255, 0, 0), 2)
    # 显示偏移量
    cv2.putText(
        frame,
        f"dx={dx}, dy={dy}",
        (30, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2,
    )


def process_detections(frame, boxes, scores, ser):
    """
    处理检测结果：绘制检测框、计算中心点偏移量、发送串口数据、画箭头和显示偏移量。

    参数:
        frame (ndarray): 当前帧图像
        boxes (ndarray): 检测框 (N,4)，格式 xyxy
        scores (ndarray): 检测框置信度
        ser (serial.Serial): 已打开的串口对象
        w0 (int): 原图宽度
        h0 (int): 原图高度
    """
    # 遍历每个检测框
    for box, score in zip(boxes, scores):
        center, offset = compute_offsets(box, frame.shape)
        draw_box_and_arrow(frame, box, score, center, offset)
        dx, dy = offset
        try:
            send_offsets(ser, dx, dy)
        except KeyboardInterrupt:
            print("停止发送")


def main():
    """
    主函数：初始化摄像头、ONNX 模型和串口，并进行实时检测和显示。

    功能:
        1. 初始化摄像头，获取视频帧
        2. 打开串口，用于发送偏移量数据给 STM32 控制舵机
        3. 加载 YOLOv5 ONNX 模型，并创建 ONNX Runtime 推理会话
        4. 循环读取摄像头帧：
            - 对帧进行预处理
            - 使用模型进行推理，得到检测框和置信度
            - 对检测结果进行置信度筛选和非极大值抑制 (NMS)
            - 绘制图像中心、检测框、箭头和偏移量
            - 将偏移量通过串口发送
            - 显示结果图像，按 'q' 退出循环
        5. 释放摄像头、关闭窗口和串口

    外部依赖:
        - OpenCV (cv2)
        - ONNX Runtime (onnxruntime)
        - pyserial (serial)

    串口说明:
        - COM_PORT: 控制舵机的串口号
        - BAUDRATE: 串口波特率
    """
    # 串口配置
    # micro-usb 是STM32->PC 的通信
    # USB-TTL 是 PC->STM32->控制舵机，也是 Nano->STM32->控制舵机, 这个 COM_PORT 是这个串口号。
    COM_PORT = r"COM43"  # 改成你实际的端口
    BAUDRATE = 115200

    # 模型路径
    onnx_path = "./models/yolov5su.onnx"
    # 置信度阈值
    conf_thres = 0.3
    # NMS IOU 阈值
    iou_thres = 0.45

    # 打开摄像头
    cap, _, _ = setup_camera(camera_id=0)
    # 打开串口
    ser = setup_serial(COM_PORT, BAUDRATE)

    # 创建 ONNX Runtime 推理会话，指定使用 CPU
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    # 获取模型输入名，用于后续推理
    input_name = session.get_inputs()[0].name

    # -----------------------------
    # 4️⃣ 循环推理
    # -----------------------------
    while True:
        ret, frame = cap.read()  # 读取一帧图像
        if not ret:
            break

        # --- Preprocess ---
        img_input = preprocess_frame(frame)

        # --- Inference ---
        boxes, scores = run_inference(session, input_name, img_input)  # 调用你写的函数
        boxes, scores = filter_boxes(
            boxes, scores, frame.shape[:2], conf_thres, iou_thres
        )

        draw_center(frame)
        process_detections(frame, boxes, scores, ser)

        # 显示结果
        cv2.imshow("ONNX YOLOv5su", frame)
        # 按 'q' 退出
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
    ser.close()
    print("串口已关闭")


if __name__ == "__main__":
    main()
