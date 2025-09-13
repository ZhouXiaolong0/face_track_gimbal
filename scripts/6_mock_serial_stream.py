#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=no-member, invalid-name

"""
simulate_tracking_data.py

模块功能：
    该模块用于通过串口周期性发送模拟偏移量数据 (dx, dy)，
    主要用于测试接收端程序、调试云台控制或仿真环境。

主要功能：
    - 打开指定串口并返回 Serial 对象
    - 循环发送随机生成的偏移量数据
    - 支持自定义发送间隔
    - 可通过 Ctrl+C 停止发送

用法示例：
    python simulate_tracking_data.py

串口配置示例：
    COM_PORT = '/dev/ttyTHS1'  # 对应 Nano 或 STM32 的串口
    BAUDRATE = 115200
"""


import sys
import time
import random
import serial


def open_serial(port: str, baudrate: int, timeout: float = 1.0) -> serial.Serial:
    """
    打开串口并返回 Serial 对象。

    参数:
        port (str): 串口号，例如 'COM3' 或 '/dev/ttyUSB0'
        baudrate (int): 波特率，例如 115200
        timeout (float, optional): 读写超时时间，单位秒，默认 1.0

    返回:
        serial.Serial: 已打开的串口对象

    异常:
        如果串口打开失败，会打印错误信息并退出程序。
    """
    try:
        ser = serial.Serial(port, baudrate, timeout=timeout)
        print(f"串口 {port} 打开成功！")
        return ser
    except serial.SerialException as e:
        print(f"打开串口失败: {e}")
        sys.exit(1)


def send_fake_offsets(ser: serial.Serial, interval: float = 1.0):
    """
    循环发送伪造偏移量数据到串口，用于调试或测试。

    参数:
        ser (serial.Serial): 已打开的串口对象
        interval (float, optional): 每次发送的时间间隔（秒），默认 1.0

    功能:
        - 在无限循环中随机生成 dx 和 dy 偏移量
        - 将偏移量构造成字符串 "dx,dy\n" 并通过串口发送
        - 打印发送的字节数和内容
        - 支持通过 Ctrl+C 停止循环发送
    """
    try:
        while True:
            # 随机生成 dx, dy
            dx = random.randint(-240, 240)
            dy = random.randint(-180, 180)

            # 构造发送字符串
            send_str = f"{dx},{dy}\n"

            # 发送
            n = ser.write(send_str.encode("ascii"))
            print(f"发送了 {n} 个字节 -> {send_str.strip()}")

            time.sleep(interval)
    except KeyboardInterrupt:
        print("停止发送")


def main():
    """
    主函数：打开串口并循环发送伪造偏移量数据。

    功能:
        1. 配置串口号和波特率（可根据实际设备修改）。
        2. 打开指定串口。
        3. 调用 send_fake_offsets 循环发送随机 dx/dy 数据。
        4. 程序退出时关闭串口并打印提示。
    """
    COM_PORT = r"COM6"  # 改成你实际的端口
    # nano
    COM_PORT = r"/dev/ttyTHS1"
    BAUDRATE = 115200

    ser = open_serial(COM_PORT, BAUDRATE)
    send_fake_offsets(ser, interval=1.0)
    ser.close()
    print("串口已关闭")


if __name__ == "__main__":
    main()
