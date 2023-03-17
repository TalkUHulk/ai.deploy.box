cmake_minimum_required(VERSION 3.19)
project(AIDeployBox)
SET(CMAKE_BUILD_TYPE "Release")

# https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB
#set(CMAKE_CXX_STANDARD 11)
#add_definitions(-std=c++11 -g -O3 -pthread)
add_definitions(-std=c++11 -pthread)


# backend options
option(ENGINE_MNN "Enable MNN" ON)
option(ENGINE_ORT "Enable ONNXRuntime" ON)
option(ENGINE_NCNN "Enable NCNN" ON)
option(ENGINE_TNN "Enable TNN" OFF)
option(ENGINE_OPV "Enable OpenVINO" ON) #Mac 与TNN冲突
option(ENGINE_PPLite "Enable paddle-lite" ON)

message(STATUS ">>>>>>>>>>>>>")
message(STATUS "AIDeployBox BUILD INFO:")
message(STATUS "\tMNN: ${ENGINE_MNN}")
message(STATUS "\tONNXRuntime: ${ENGINE_ONNXRUNTIME}")
message(STATUS "\tNCNN: ${ENGINE_NCNN}")
message(STATUS "\tTNN: ${ENGINE_TNN}")
message(STATUS "\tOpenVINO: ${ENGINE_OPV}")
message(STATUS "\tPaddleLite: ${ENGINE_PPLite}")


add_subdirectory(source)
add_subdirectory(samples)

