cmake_minimum_required(VERSION 3.16)
project(AIDeployBox)
SET(CMAKE_BUILD_TYPE "Release")

# https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB
#set(CMAKE_CXX_STANDARD 11)
#add_definitions(-std=c++11 -g -O3 -pthread)
#add_definitions(-std=c++11 -pthread)



# backend options
option(ENGINE_MNN "Enable MNN" OFF)
option(ENGINE_ORT "Enable ONNXRuntime" OFF)
option(ENGINE_NCNN "Enable NCNN" OFF)
option(ENGINE_TNN "Enable TNN" OFF)
option(ENGINE_OPV "Enable OpenVINO" OFF) #Mac 与TNN冲突
option(ENGINE_PPLite "Enable paddle-lite" OFF)
option(ENGINE_NCNN_WASM "Enable NCNN WASM" OFF)


message(STATUS ">>>>>>>>>>>>>")
message(STATUS "AIDeployBox BUILD INFO:")
message(STATUS "\tMNN: ${ENGINE_MNN}")
message(STATUS "\tONNXRuntime: ${ENGINE_ORT}")
message(STATUS "\tNCNN: ${ENGINE_NCNN}")
message(STATUS "\tTNN: ${ENGINE_TNN}")
message(STATUS "\tOpenVINO: ${ENGINE_OPV}")
message(STATUS "\tPaddleLite: ${ENGINE_PPLite}")
message(STATUS "\tWASM: ${ENGINE_NCNN_WASM}")

if(ENGINE_NCNN_WASM)
    if(NOT WASM_FEATURE)
        message(FATAL_ERROR "You must pass cmake option -DWASM_FEATURE and possible values are basic, simd, threads and simd-threads")
    endif()
    set(OS wasm)
#    option(EIGEN_DONT_PARALLELIZE "Eigen OMP" ON)
elseif (UNIX AND NOT APPLE)
    set(OS linux)
elseif (APPLE)
    set(OS mac)
else ()
    set(OS win64)
endif ()

set(PLATFORM ${OS} CACHE INTERNAL "current os" )

add_subdirectory(source)

if(NOT ENGINE_NCNN_WASM)
    add_definitions(-std=c++11 -pthread)
    add_subdirectory(samples)
else ()
    add_definitions(-std=c++11)
    add_subdirectory(sample-wasm)
endif ()

