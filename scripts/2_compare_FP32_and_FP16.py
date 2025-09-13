#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=no-member

"""
2_compare_FP32_and_FP16.py

功能：
- 对比 YOLOv5 模型在 FP32 与 FP16 精度下的文件大小、推理性能和检测结果差异。
- 性能指标包括：
    - 平均推理时间（ms）
    - FPS（Frames Per Second）
    - CPU 占用
    - GPU 显存使用
- 检测结果差异指标包括：
    - 检测框中心点偏差
    - 检测框置信度差异

使用场景：
- 模型精度/性能评估
- 推理优化分析
- 部署前模型选择参考
"""

import os
import time
import psutil
import cv2
import torch
import numpy as np
from ultralytics import YOLO


def print_model_size(fp32_path: str, fp16_path: str) -> None:
    """
    打印 FP32 和 FP16 模型的文件大小。

    参数
    ----
    fp32_path : str
        FP32 模型文件路径。
    fp16_path : str
        FP16 模型文件路径。
    """
    print(f"FP32 模型大小为：{os.path.getsize(fp32_path)/1024**2:.2f} MB")
    print(f"FP16 模型大小为：{os.path.getsize(fp16_path)/1024**2:.2f} MB")


def load_yolo_model(model_path: str, use_fp16: bool = False) -> YOLO:
    """
    加载 YOLOv5 模型并设置 FP32/FP16 精度。

    参数
    ----
    model_path : str
        YOLOv5 模型权重文件路径。
    use_fp16 : bool
        是否使用 FP16 精度。默认为 False。

    返回
    ----
    YOLO
        已加载的 YOLO 模型对象。
    """
    model = YOLO(model_path)
    if use_fp16:
        model.model.half()
    else:
        model.model.float()
    return model


def get_sample_frame() -> np.ndarray:
    """
    从摄像头获取一帧图像。

    返回
    ----
    np.ndarray
        BGR 格式的图像数组。

    异常
    ----
    RuntimeError
        如果无法读取摄像头图像。
    """
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("无法读取摄像头图像")
    return frame


def benchmark_model(model, frame, repeat=50):
    """
    对指定 YOLO 模型进行推理性能测试。

    参数
    ----
    model : YOLO
        YOLO 模型对象。
    frame : np.ndarray
        输入图像。
    repeat : int
        重复推理次数，用于计算平均性能指标。默认 50。

    返回
    ----
    tuple
        avg_time : float
            平均推理时间（秒）。
        avg_fps : float
            平均 FPS。
        avg_cpu : float
            平均 CPU 占用百分比。
        gpu_mem : float
            GPU 显存使用量（MB）。
        results : YOLO 输出结果
            模型最后一次推理结果。
    """
    torch.cuda.empty_cache()  # 清理 GPU 缓存，避免影响显存统计
    times = []
    cpu_usages = []

    for _ in range(repeat):
        start_cpu = psutil.cpu_percent(interval=None)  # 记录 CPU 使用率
        start_time = time.time()  # 记录开始时间
        # results = model(frame)
        results = model(frame, verbose=False)  # 推理
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = time.time()
        end_cpu = psutil.cpu_percent(interval=None)

        times.append(end_time - start_time)  # 记录本次推理耗时
        cpu_usages.append(end_cpu - start_cpu)  # 记录本次 CPU 占用

    avg_time = sum(times) / len(times)  # 平均推理时间
    avg_fps = 1 / avg_time  # 平均 FPS = 1 / 平均单帧耗时
    avg_cpu = sum(cpu_usages) / len(cpu_usages)  # 平均 CPU 占用

    gpu_mem = (
        torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
    )
    return avg_time, avg_fps, avg_cpu, gpu_mem, results


def collect_frames(num_frames: int = 50) -> list:
    """
    从摄像头采集多帧图像。

    参数
    ----
    num_frames : int
        需要采集的帧数，默认 50。

    返回
    ----
    list
        包含采集到的 BGR 图像的列表。
    """
    cap = cv2.VideoCapture(0)
    frames = []

    print(f"正在采集 {num_frames} 帧...")
    for i in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            print(f"第{i}帧无法读取，跳过")
            continue
        frames.append(frame)
    cap.release()
    print(f"采集完成，总共 {len(frames)} 帧可用")
    return frames


def compare_detection_results(frames: list, model_fp32: YOLO, model_fp16: YOLO):
    """
    对多帧图像进行 FP32 与 FP16 模型检测结果对比。

    参数
    ----
    frames : list
        多帧 BGR 图像列表。
    model_fp32 : YOLO
        FP32 模型对象。
    model_fp16 : YOLO
        FP16 模型对象。

    输出
    ----
    - 检测框中心点偏差平均值及方差
    - 检测框置信度差平均值及方差
    """
    dx_list, dy_list, dc_list = [], [], []

    for _, frame_in in enumerate(frames):
        # 推理
        res_fp32 = model_fp32(frame_in, verbose=False)
        res_fp16 = model_fp16(frame_in, verbose=False)

        # 取第一个目标或多个目标
        boxes1 = res_fp32[0].boxes.xywh.cpu().numpy()
        boxes2 = res_fp16[0].boxes.xywh.cpu().numpy()
        conf1 = res_fp32[0].boxes.conf.cpu().numpy()
        conf2 = res_fp16[0].boxes.conf.cpu().numpy()

        n = min(len(boxes1), len(boxes2))
        for i in range(n):
            dx_list.append(boxes2[i, 0] - boxes1[i, 0])
            dy_list.append(boxes2[i, 1] - boxes1[i, 1])
            dc_list.append(conf2[i] - conf1[i])

    # 4. 输出平均值和方差
    dx_mean, dy_mean, dc_mean = np.mean(dx_list), np.mean(dy_list), np.mean(dc_list)
    dx_std, dy_std, dc_std = np.std(dx_list), np.std(dy_list), np.std(dc_list)

    print("\n=== FP16 vs FP32 多帧对比 ===")
    print(f"中心点偏差平均 dx={dx_mean:.2f}, dy={dy_mean:.2f}")
    print(f"中心点偏差方差 dx={dx_std:.2f}, dy={dy_std:.2f}")
    print(f"置信度差平均 dc={dc_mean:.3f}, 方差 dc_std={dc_std:.3f}")


def main():
    """
    主程序入口：执行 FP32 与 FP16 YOLOv5 模型的对比分析。

    功能步骤：
    1. 输出 FP32 与 FP16 模型文件大小。
    2. 加载 FP32 和 FP16 模型。
    3. 获取单帧图像进行性能 benchmark。
    4. 测试并打印平均推理时间、FPS、CPU 和 GPU 占用。
    5. 采集多帧图像，对比检测框中心点偏差及置信度差。
    """
    fp32_path = "./models/model_fp32.pt"
    fp16_path = "./models/model_fp16.pt"
    model_file_path = "./models/yolov5su.pt"

    # 模型文件大小对比
    print_model_size(fp32_path, fp16_path)

    # 加载模型
    model_fp32 = load_yolo_model(model_file_path, use_fp16=False)
    model_fp16 = load_yolo_model(model_file_path, use_fp16=True)

    # 获取样本帧
    sample_frame = get_sample_frame()

    # benchmark 性能
    print("\nBenchmarking FP32...")
    fp32_time, fp32_fps, fp32_cpu, fp32_gpu, _ = benchmark_model(
        model_fp32, sample_frame
    )
    print(
        f"FP32 - 推理时间: {fp32_time*1000:.2f} ms,"
        f" FPS(Frames Per Second): {fp32_fps:.2f},"
        f" CPU: {fp32_cpu:.1f}%,"
        f" GPU显存: {fp32_gpu:.2f} MB"
    )

    print("\nBenchmarking FP16...")
    fp16_time, fp16_fps, fp16_cpu, fp16_gpu, _ = benchmark_model(
        model_fp16, sample_frame
    )
    print(
        f"FP16 - 推理时间: {fp16_time*1000:.2f} ms,"
        f" FPS(Frames Per Second): {fp16_fps:.2f},"
        f" CPU: {fp16_cpu:.1f}%,"
        f" GPU显存: {fp16_gpu:.2f} MB"
    )

    # 采集多帧进行检测结果对比
    frames = collect_frames(num_frames=50)
    compare_detection_results(frames, model_fp32, model_fp16)


if __name__ == "__main__":
    main()
