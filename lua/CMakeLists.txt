cmake_minimum_required(VERSION 3.16)
project(pyAiDB CXX C)
SET(CMAKE_CXX_STANDARD 14)
SET(CMAKE_CXX_STANDARD_REQUIRED ON)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
SET(CMAKE_BUILD_TYPE "Release")


find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

include_directories(${CMAKE_CURRENT_LIST_DIR}/../source)
include_directories(${CMAKE_CURRENT_LIST_DIR}/../source/core)
include_directories(${CMAKE_CURRENT_LIST_DIR}/../3rdparty)


set(LIB_PATH ${CMAKE_CURRENT_LIST_DIR}/../libs/${PLATFORM})
link_directories(${LIB_PATH})
link_directories(${LIB_PATH}/tnn/)
link_directories(${LIB_PATH}/paddlelite/)
link_directories(${LIB_PATH}/openvino/intel64)

#include_directories( /usr/local/opt/lua@5.3/include/lua5.3)
#link_directories(/usr/local/opt/lua@5.3/lib)

find_package(Lua REQUIRED)

include_directories(${LUA_INCLUDE_DIR})

add_library(AiDBLua SHARED ${SOURCE_FILES} ${CMAKE_CURRENT_SOURCE_DIR}/wrapper.cpp)
#target_link_libraries(AiDBLua ${OpenCV_LIBS} ${LINK_LIBS} liblua.5.3.dylib AiDB_C)
target_link_libraries(AiDBLua ${OpenCV_LIBS} ${LINK_LIBS} ${LUA_LIBRARIES} AiDB_C)














