cmake_minimum_required(VERSION 3.18.0)

# 项目名称和 C++ 标准
project(FaceTrackGimbal LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")

# 包含头文件目录
include_directories(include)

# 自动查找所有源文件和头文件
file(GLOB_RECURSE SOURCES "src/*.cpp")
file(GLOB_RECURSE HEADERS "include/*.h")

# ======== OpenCV =========
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# ======== ONNX Runtime =========
set(ONNXRUNTIME_DIR /home/jetson/onnxruntime)

include_directories(${ONNXRUNTIME_DIR}/include/onnxruntime/core/session)

find_library(ONNXRUNTIME_LIB onnxruntime
    PATHS ${ONNXRUNTIME_DIR}/build/Linux/Release
)

if (NOT ONNXRUNTIME_LIB)
    message(FATAL_ERROR "onnxruntime library not found!")
endif()

# ======== Threads (pthread) =========
find_package(Threads REQUIRED)

# =================================

# 生成可执行文件
add_executable(${PROJECT_NAME}
    ${SOURCES}
    ${HEADERS}
)

# ========= 链接 OpenCV / ONNX Runtime / Threads ===========
target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBS}
    ${ONNXRUNTIME_LIB}
    Threads::Threads
)
