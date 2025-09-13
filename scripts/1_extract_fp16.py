#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=no-member

"""
extract_fp16.py

功能：
- 将 YOLOv5 模型的权重导出为 FP32 和 FP16 两种精度。
- 生成的权重可用于训练加速、部署或转换为 ONNX/TensorRT 模型。
- 支持指定保存路径。

主要函数：
- export_model_weights(model_path: str, fp32_path: str, fp16_path: str)
    根据给定的 YOLOv5 模型路径导出 FP32 和 FP16 权重文件。
- main()
    主程序入口，用于执行 FP32/FP16 权重导出操作。

注意事项：
- 需要已安装 ultralytics 和 torch 库。
- 导出的权重文件路径需确保可写权限。
- FP16 权重适合在支持半精度计算的 GPU 上推理或训练。
"""

import torch
from ultralytics import YOLO


def export_model_weights(model_path: str, fp32_path: str, fp16_path: str) -> None:
    """
    导出 YOLOv5 模型的 FP32 和 FP16 权重文件。

    参数
    ----
    model_path : str
        原始 YOLOv5 PyTorch 模型路径（.pt 文件）。
    fp32_path : str
        导出 FP32 权重的保存路径。
    fp16_path : str
        导出 FP16 权重的保存路径。

    返回
    ----
    None
        导出完成后，会在指定路径生成 FP32 和 FP16 权重文件。

    注意
    ----
    - FP32 权重使用 model.model.float() 保存。
    - FP16 权重使用 model.model.half() 保存。
    - 保存前会覆盖同名文件，请确保备份重要数据。
    """
    # 加载模型
    model = YOLO(model_path)

    # 保存 FP32 权重
    model.model.float()
    torch.save(model.model.state_dict(), fp32_path)
    print(f"FP32 权重已保存到 {fp32_path}")

    # 保存 FP16 权重
    model.model.half()
    torch.save(model.model.state_dict(), fp16_path)
    print(f"FP16 权重已保存到 {fp16_path}")


def main():
    """
    主函数入口，执行 YOLOv5 FP32 和 FP16 权重导出。

    功能：
    1. 指定原始模型路径。
    2. 指定 FP32 和 FP16 权重的保存路径。
    3. 调用 export_model_weights() 完成权重导出。
    """

    FP32_PATH = "./models/model_fp32.pt"
    FP16_PATH = "./models/model_fp16.pt"
    MODEL_PATH = "./models/yolov5su.pt"

    export_model_weights(MODEL_PATH, FP32_PATH, FP16_PATH)


if __name__ == "__main__":
    main()
