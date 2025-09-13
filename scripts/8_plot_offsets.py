"""
这个脚本用于读取 CSV 文件中的云台偏移量数据，
并使用 matplotlib 绘制原始偏移量与滤波后偏移量的对比折线图。
"""

# pylint: disable=invalid-name

import pandas as pd
import matplotlib.pyplot as plt


def main():
    """
    读取 offsets.csv 文件，将滤波后的偏移量取整，
    并绘制原始偏移量和滤波后偏移量在 dx、dy 两个维度的对比折线图。
    """
    # 读取 CSV 文件（替换为你的实际路径）
    df = pd.read_csv(
        "data/offsets.csv",
        header=None,
        names=["raw_dx", "raw_dy", "filtered_dx", "filtered_dy"],
    )

    # 取整（后两列）
    df["filtered_dx"] = df["filtered_dx"].astype(int)
    df["filtered_dy"] = df["filtered_dy"].astype(int)

    # 绘制折线图
    plt.figure(figsize=(12, 6))

    # dx 对比
    plt.subplot(2, 1, 1)
    plt.plot(df.index, df["raw_dx"], label="Raw dx", linestyle="--")
    plt.plot(df.index, df["filtered_dx"], label="Filtered dx", linestyle="-")
    plt.title("DX Comparison")
    plt.xlabel("Frame")
    plt.ylabel("dx")
    plt.legend()
    plt.grid(True)

    # dy 对比
    plt.subplot(2, 1, 2)
    plt.plot(df.index, df["raw_dy"], label="Raw dy", linestyle="--")
    plt.plot(df.index, df["filtered_dy"], label="Filtered dy", linestyle="-")
    plt.title("DY Comparison")
    plt.xlabel("Frame")
    plt.ylabel("dy")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
