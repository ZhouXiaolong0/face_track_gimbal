#!/bin/bash

# 创建 build 目录
mkdir -p build
cd build || exit 1

# 生成 Makefile
cmake ..

# 编译
make -j$(nproc)

# 运行
./FaceTrackGimbal
