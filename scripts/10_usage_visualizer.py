"""
usage_visualizer.py

本模块用于可视化系统资源使用情况。
它从 CSV 文件中读取 CPU、内存和 GPU 使用率，并使用折线图进行展示，
方便观察资源的变化趋势。

CSV 文件格式示例：
    cpu_usage,mem_usage,gpu_usage
    0.0,59.5,4.2
    47.35,62.56,10.8
    ...

"""

# flake8: noqa: E501
# pylint: disable=invalid-name

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams["font.sans-serif"] = ["SimHei"]  # 或 ['Microsoft YaHei']
matplotlib.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题


def visualize_usage(csv_file: str) -> None:
    """
    从 CSV 文件中读取 CPU、内存和 GPU 使用率数据并进行可视化。

    参数:
        csv_file (str): 包含使用率统计数据的 CSV 文件路径，
                        文件需包含 'cpu_usage', 'mem_usage', 'gpu_usage' 三列。

    返回:
        None: 分别显示三个 matplotlib 折线图。
    """
    # 读取 CSV 数据
    data = pd.read_csv(csv_file)

    # CPU 使用率
    plt.figure(figsize=(8, 4))
    plt.plot(data["cpu_usage"], label="CPU 使用率 (%)", marker="o", color="tab:blue")
    plt.title("CPU 使用率变化趋势")
    plt.xlabel("采样点序号")
    plt.ylabel("使用率 (%)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # 内存使用率
    plt.figure(figsize=(8, 4))
    plt.plot(data["mem_usage"], label="内存使用率 (%)", marker="s", color="tab:green")
    plt.title("内存使用率变化趋势")
    plt.xlabel("采样点序号")
    plt.ylabel("使用率 (%)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # GPU 使用率
    plt.figure(figsize=(8, 4))
    plt.plot(data["gpu_usage"], label="GPU 使用率 (%)", marker="^", color="tab:red")
    plt.title("GPU 使用率变化趋势")
    plt.xlabel("采样点序号")
    plt.ylabel("使用率 (%)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # 显示所有图
    plt.show()


def main():
    """主函数，运行资源使用率可视化程序。"""
    csv_file = "data/system_info.csv"  # 修改为你的 CSV 文件名
    visualize_usage(csv_file)


if __name__ == "__main__":
    main()
