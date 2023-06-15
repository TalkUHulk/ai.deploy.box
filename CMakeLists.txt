cmake_minimum_required(VERSION 3.16)
project(AiDB)
SET(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_STANDARD 14)

#add_definitions(-std=c++11 -g -O3 -pthread)
#add_definitions(-std=c++11 -pthread)



# backend options
option(ENGINE_MNN "Enable MNN" ON)
option(ENGINE_ORT "Enable ONNXRuntime" OFF)
option(ENGINE_NCNN "Enable NCNN" OFF)
option(ENGINE_TNN "Enable TNN" OFF)
option(ENGINE_OPV "Enable OpenVINO" OFF) #Mac 与TNN冲突
option(ENGINE_PPLite "Enable paddle-lite" OFF)
option(ENGINE_NCNN_WASM "Enable NCNN WASM" OFF)
option(BUILD_SAMPLE "Enable build sample" OFF)
option(BUILD_PYTHON "Enable python api" OFF)
option(OPENCV_HAS_FREETYPE "opencv-contrib complied with freetype or not. cv::putText can put chinese." OFF)

message(STATUS ">>>>>>>>>>>>>")
message(STATUS "AiDB BUILD INFO:")
message(STATUS "\tMNN: ${ENGINE_MNN}")
message(STATUS "\tONNXRuntime: ${ENGINE_ORT}")
message(STATUS "\tNCNN: ${ENGINE_NCNN}")
message(STATUS "\tTNN: ${ENGINE_TNN}")
message(STATUS "\tOpenVINO: ${ENGINE_OPV}")
message(STATUS "\tPaddleLite: ${ENGINE_PPLite}")
message(STATUS "\tWASM: ${ENGINE_NCNN_WASM}")
message(STATUS "\tBuild Sample: ${BUILD_SAMPLE}")
message(STATUS "\tBuild Python: ${BUILD_PYTHON}")
message(STATUS "\tOPENCV_HAS_FREETYPE: ${OPENCV_HAS_FREETYPE}")

add_definitions(-DOPENCV_HAS_FREETYPE)

message("=========>>>> ${CMAKE_BINARY_DIR}")
if(ENGINE_NCNN_WASM)
    if(NOT WASM_FEATURE)
        message(FATAL_ERROR "You must pass cmake option -DWASM_FEATURE and possible values are basic, simd, threads and simd-threads")
    endif()
    set(OS wasm)
    set(BUILD_SAMPLE OFF)
elseif (UNIX AND NOT APPLE)
    set(OS linux)
elseif (APPLE)
    set(OS mac)
elseif (ANDROID)
    set(OS android)
    set(BUILD_SAMPLE OFF)
else ()
    set(OS win64)
endif ()

set(PLATFORM ${OS} CACHE INTERNAL "PlatformOS" )

add_subdirectory(source)

if(NOT ENGINE_NCNN_WASM)
    add_definitions(-pthread)
endif ()

if(ENGINE_NCNN_WASM)
    add_subdirectory(sample-wasm)
endif ()

if(BUILD_SAMPLE)
    add_subdirectory(samples)
endif ()

if(BUILD_PYTHON)
    add_subdirectory(python)
endif ()
