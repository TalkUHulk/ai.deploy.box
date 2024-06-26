cmake_minimum_required(VERSION 3.19)
set(ALL_TARGETS "" CACHE INTERNAL "")

project(AiDB)
SET(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_STANDARD 17)

#add_definitions(-std=c++11 -g -O3 -pthread)
#add_definitions(-std=c++11 -pthread)


# backend options
option(ENGINE_MNN "Enable MNN" OFF)
option(ENGINE_ORT "Enable ONNXRuntime" ON)
option(ENGINE_NCNN "Enable NCNN" OFF)
option(ENGINE_TNN "Enable TNN" OFF)
option(ENGINE_OPV "Enable OpenVINO" OFF) #Mac 与TNN冲突
option(ENGINE_PPLite "Enable paddle-lite" OFF)
option(ENGINE_NCNN_WASM "Enable NCNN WASM" OFF)
option(BUILD_SAMPLE "Enable build sample" ON)
option(BUILD_PYTHON "Enable python api" OFF)
option(BUILD_LUA "Enable lua api" OFF)
option(OPENCV_HAS_FREETYPE "opencv-contrib complied with freetype or not. cv::putText can put chinese." OFF)
option(C_API "open c api" OFF)
option(ENABLE_SD "open stable diffusion" OFF) # Experimental

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
message(STATUS "\tBuild C_API: ${C_API}")
message(STATUS "\tOPENCV_HAS_FREETYPE: ${OPENCV_HAS_FREETYPE}")

include("${CMAKE_CURRENT_SOURCE_DIR}/tools/aidbCommon.cmake")

if(ENABLE_SD AND ENGINE_ORT)
    add_definitions(-DSD_ON)
    find_package(Protobuf REQUIRED)
endif ()


if(OPENCV_HAS_FREETYPE)
    add_definitions(-DOPENCV_HAS_FREETYPE)
endif()

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

message("-- AiDB DEVICE INFO: CPU: ${CMAKE_HOST_SYSTEM_PROCESSOR} OS: ${OS}")

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

if(BUILD_LUA)
    add_subdirectory(lua)
endif ()



add_custom_target(BUILD_SUCCESSFUL ALL
        DEPENDS ${ALL_TARGETS}
        COMMAND ${CMAKE_COMMAND} -E echo ""
        COMMAND ${CMAKE_COMMAND} -E echo "=========================================================================="
        COMMAND ${CMAKE_COMMAND} -E echo "                           Compile complete!                              "
        COMMAND ${CMAKE_COMMAND} -E echo ""
        COMMAND ${CMAKE_COMMAND} -E echo ""
        COMMAND ${CMAKE_COMMAND} -E echo "███████╗███╗   ██╗     ██╗ ██████╗ ██╗   ██╗     █████╗ ██╗██████╗ ██████╗ "
        COMMAND ${CMAKE_COMMAND} -E echo "██╔════╝████╗  ██║     ██║██╔═══██╗╚██╗ ██╔╝    ██╔══██╗██║██╔══██╗██╔══██╗"
        COMMAND ${CMAKE_COMMAND} -E echo "█████╗  ██╔██╗ ██║     ██║██║   ██║ ╚████╔╝     ███████║██║██║  ██║██████╔╝"
        COMMAND ${CMAKE_COMMAND} -E echo "██╔══╝  ██║╚██╗██║██   ██║██║   ██║  ╚██╔╝      ██╔══██║██║██║  ██║██╔══██╗"
        COMMAND ${CMAKE_COMMAND} -E echo "███████╗██║ ╚████║╚█████╔╝╚██████╔╝   ██║       ██║  ██║██║██████╔╝██████╔╝"
        COMMAND ${CMAKE_COMMAND} -E echo "╚══════╝╚═╝  ╚═══╝ ╚════╝  ╚═════╝    ╚═╝       ╚═╝  ╚═╝╚═╝╚═════╝ ╚═════╝ "
        COMMAND ${CMAKE_COMMAND} -E echo "==========================================================================="
        COMMAND ${CMAKE_COMMAND} -E echo ""
        )


