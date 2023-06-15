cmake_minimum_required(VERSION 3.16)
project(pyAiDB CXX C)
SET(CMAKE_CXX_STANDARD 14)
SET(CMAKE_CXX_STANDARD_REQUIRED ON)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
SET(CMAKE_BUILD_TYPE "Release")

#set(PYTHONVERSION "3.6.12")
set(LIB_VERSION_INFO 1.0.0)



find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

include_directories(${CMAKE_CURRENT_LIST_DIR}/../source)
include_directories(${CMAKE_CURRENT_LIST_DIR}/../source/core)
include_directories(${CMAKE_CURRENT_LIST_DIR}/../3rdparty)
include_directories(${CMAKE_CURRENT_LIST_DIR}/wrapper)

set(LIB_PATH ${CMAKE_CURRENT_LIST_DIR}/../libs/${PLATFORM})
link_directories(${LIB_PATH})
link_directories(${LIB_PATH}/tnn/)
link_directories(${LIB_PATH}/paddlelite/)
link_directories(${LIB_PATH}/openvino/intel64)

add_subdirectory(./pybind11)
pybind11_add_module(pyAiDB ${PROJECT_SOURCE_DIR}/wrapper.cpp)

target_link_libraries(pyAiDB PRIVATE AiDB ${OpenCV_LIBS})
target_compile_definitions(pyAiDB PRIVATE VERSION_INFO=${LIB_VERSION_INFO})

#if (UNIX AND NOT APPLE)
#    add_custom_command(
#            TARGET pyAiDB
#            PRE_BUILD
#            COMMAND ${CMAKE_COMMAND} -E copy ${PROJECT_SOURCE_DIR}/../3rdparty/lib/linux/libAiDB.so ${CMAKE_BINARY_DIR}/python/
#    )
#elseif (APPLE)
#    add_custom_command(
#            TARGET pyAiDB
#            PRE_BUILD
#            COMMAND ${CMAKE_COMMAND} -E copy  ${CMAKE_BINARY_DIR}/source/libAiDB.dylib ${CMAKE_BINARY_DIR}/python/
#    )
#endif()

