include_directories(${CMAKE_CURRENT_LIST_DIR}/../3rdparty/
        ${CMAKE_CURRENT_LIST_DIR}/core
        ${CMAKE_CURRENT_LIST_DIR})

file(GLOB_RECURSE CORE_SOURCE_FILES ${CMAKE_CURRENT_LIST_DIR}/core/*.cpp)
file(GLOB_RECURSE YAML_CPP_SOURCE_FILES
        ${CMAKE_CURRENT_LIST_DIR}/../3rdparty/yaml-cpp/*.cpp
        ${CMAKE_CURRENT_LIST_DIR}/../3rdparty/yaml-cpp/*.cc)
file(GLOB_RECURSE UTILS_SOURCE_FILES ${CMAKE_CURRENT_LIST_DIR}/utils/*.cpp)

set(SOURCE_FILES)
list(APPEND SOURCE_FILES ${CORE_SOURCE_FILES})
list(APPEND SOURCE_FILES ${YAML_CPP_SOURCE_FILES})
list(APPEND SOURCE_FILES ${UTILS_SOURCE_FILES})



IF(ENGINE_MNN)
    # 2.3.0
    add_definitions(-DENABLE_MNN)
    file(GLOB_RECURSE MNN_SOURCE_FILES ${CMAKE_CURRENT_LIST_DIR}/backend/mnn/*.cpp)
    list(APPEND SOURCE_FILES ${MNN_SOURCE_FILES})
    list(APPEND LINK_LIBS MNN)
ENDIF()

IF(ENGINE_ORT)
    add_definitions(-DENABLE_ORT)
    file(GLOB_RECURSE ONNXRUNTIME_SOURCE_FILES ${CMAKE_CURRENT_LIST_DIR}/backend/onnxruntime/*.cpp)
    list(APPEND SOURCE_FILES ${ONNXRUNTIME_SOURCE_FILES})
    list(APPEND LINK_LIBS onnxruntime)
ENDIF()

IF(ENGINE_NCNN)
    add_definitions(-DENABLE_NCNN)
    file(GLOB_RECURSE NCNN_SOURCE_FILES ${CMAKE_CURRENT_LIST_DIR}/backend/ncnn/*.cpp)
    list(APPEND SOURCE_FILES ${NCNN_SOURCE_FILES})
    list(APPEND LINK_LIBS ncnn)
ENDIF()

IF(ENGINE_TNN)
    add_definitions(-DENABLE_TNN)
    file(GLOB_RECURSE TNN_SOURCE_FILES ${CMAKE_CURRENT_LIST_DIR}/backend/tnn/*.cpp)
    list(APPEND SOURCE_FILES ${TNN_SOURCE_FILES})
    list(APPEND LINK_LIBS TNN)
ENDIF()


IF(ENGINE_OPV)
    add_definitions(-DENABLE_OPV)
    include_directories(${CMAKE_CURRENT_LIST_DIR}/../3rdparty/openvino)
    include_directories(${CMAKE_CURRENT_LIST_DIR}/../3rdparty/openvino/ie)
    include_directories(${CMAKE_CURRENT_LIST_DIR}/../3rdparty/openvino/ngraph)
    link_directories(${CMAKE_CURRENT_LIST_DIR}/../libs/${PLATFORM}/openvino/intel64/)

    file(GLOB_RECURSE OPV_SOURCE_FILES ${CMAKE_CURRENT_LIST_DIR}/backend/openvino/*.cpp)
    list(APPEND SOURCE_FILES ${OPV_SOURCE_FILES})
    list(APPEND LINK_LIBS openvino)

ENDIF()

IF(ENGINE_PPLite)
    add_definitions(-DENABLE_PPLite)
    include_directories(${CMAKE_CURRENT_LIST_DIR}/../3rdparty/PaddleLite)
    include_directories(${CMAKE_CURRENT_LIST_DIR}/../3rdparty/mklml/include)

    file(GLOB_RECURSE PPLite_SOURCE_FILES ${CMAKE_CURRENT_LIST_DIR}/backend/paddle_lite/*.cpp)
    list(APPEND SOURCE_FILES ${PPLite_SOURCE_FILES})
#    list(APPEND LINK_LIBS paddle_light_api_shared iomp5 dl)
    list(APPEND LINK_LIBS paddle_full_api_shared iomp5 dl)
ENDIF()

#if (UNIX AND NOT APPLE)
#    set(LIB_PATH ${CMAKE_CURRENT_LIST_DIR}/../libs/linux)
#    #    set(YAML yaml-cpp.0.7)
#    set(PLATFORM Linux)
#elseif (APPLE)
#    set(LIB_PATH ${CMAKE_CURRENT_LIST_DIR}/../libs/mac)
#    #    set(YAML yaml-cpp.0.7)
#    set(PLATFORM MacOs)
#else ()
#    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS TRUE)
#    set(LIB_PATH ${CMAKE_CURRENT_LIST_DIR}/../libs/x64)
#    #    set(YAML yaml-cpp)
#    set(PLATFORM Windows)
#endif ()

#message("*************** ${PLATFORM}")

#if (${PLATFORM} STREQUAL linux)
#    set(LIB_PATH ${CMAKE_CURRENT_LIST_DIR}/../libs/linux)
#elseif (${PLATFORM} STREQUAL MacOs)
#    set(LIB_PATH ${CMAKE_CURRENT_LIST_DIR}/../libs/mac)
#else ()
#    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS TRUE)
#    set(LIB_PATH ${CMAKE_CURRENT_LIST_DIR}/../libs/win64)
#endif ()

if (${PLATFORM} STREQUAL win64)
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS TRUE)
endif ()

set(LIB_PATH ${CMAKE_CURRENT_LIST_DIR}/../libs/${PLATFORM})
link_directories(${LIB_PATH})

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

file(GLOB_RECURSE INPUT_SOURCE_FILES ${CMAKE_CURRENT_LIST_DIR}/preprocess/*.cpp)
file(GLOB_RECURSE OUTPUT_SOURCE_FILES ${CMAKE_CURRENT_LIST_DIR}/utility/*.cpp)

list(APPEND SOURCE_FILES ${INPUT_SOURCE_FILES} ${OUTPUT_SOURCE_FILES})

add_library(aidb SHARED ${SOURCE_FILES})
target_link_libraries(aidb ${OpenCV_LIBS} ${LINK_LIBS})


set(INCLUDE_FILES
        ${CMAKE_CURRENT_LIST_DIR}/core/AIDBData.h
        ${CMAKE_CURRENT_LIST_DIR}/core/AIDBInput.h
        ${CMAKE_CURRENT_LIST_DIR}/core/Interpreter.h
        ${CMAKE_CURRENT_LIST_DIR}/core/StatusCode.h
        ${CMAKE_CURRENT_LIST_DIR}/utility/Utility.h)

install(FILES ${INCLUDE_FILES}
        DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../include)



