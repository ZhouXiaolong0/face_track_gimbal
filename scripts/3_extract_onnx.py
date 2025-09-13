#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=no-member

"""
extract_onnx.py

功能：
- 将 YOLOv5 模型导出为 ONNX 格式，支持 FP16 和 FP32 两种精度。
- 导出的 ONNX 文件可用于 ONNX Runtime、TensorRT 或其他推理框架部署。
- 支持动态输入尺寸，便于不同分辨率图像推理。

模块内容：
- export_onnx(model_path: str, fp16: bool = True)
    根据指定的 PyTorch YOLOv5 模型路径导出 ONNX 文件，可选择 FP16 或 FP32 精度。
- main()
    主程序入口，用于执行 ONNX 导出操作。

注意事项：
- 需要安装 ultralytics 库（YOLOv5）。
- 导出的 ONNX 文件默认保存在当前工作目录，文件名根据精度自动命名：
    - FP16: yolov5su_fp16.onnx
    - FP32: yolov5su_fp32.onnx
- 动态输入尺寸已启用。
"""

from ultralytics import YOLO


def export_onnx(model_path: str, fp16: bool = True) -> None:
    """
    导出 YOLOv5 模型为 ONNX 文件。

    参数
    ----
    model_path : str
        PyTorch YOLOv5 模型文件路径（.pt）。
    fp16 : bool, 可选
        是否导出为 FP16 精度，默认为 True。
        如果为 False，则导出 FP32 精度。

    返回
    ----
    None
        导出完成后将在当前工作目录生成 ONNX 文件。
        文件名根据精度自动命名：
        - FP16: yolov5su_fp16.onnx
        - FP32: yolov5su_fp32.onnx

    注意
    ----
    - 动态输入尺寸已启用，支持不同分辨率的推理。
    - 导出过程会对模型进行简化以优化推理速度。
    """
    model = YOLO(model_path)
    filename = "yolov5su_fp16.onnx" if fp16 else "yolov5su_fp32.onnx"
    model.export(format="onnx", opset=12, half=fp16, dynamic=True, simplify=True)
    print(f"ONNX 模型已导出: {filename}")


def main():
    """
    主函数入口，执行 ONNX 模型导出操作。

    功能：
    1. 指定 PyTorch YOLOv5 模型路径。
    2. 调用 export_onnx() 导出 ONNX 文件（默认 FP16，可选择 FP32）。
    """

    MODEL_PATH = "./models/yolov5su.pt"

    export_onnx(MODEL_PATH, fp16=True)
    # 如果需要导出 FP32，可以取消下面注释
    # export_onnx(MODEL_PATH, fp16=False)


if __name__ == "__main__":
    main()
