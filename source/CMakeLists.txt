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
    if(ANDROID)
        #        link_directories(${CMAKE_CURRENT_LIST_DIR}/../libs/${PLATFORM}/${ANDROID_ABI}/)
        add_library(MNN SHARED IMPORTED)
        add_library(MNN_CL SHARED IMPORTED)
        add_library(MNN_Express SHARED IMPORTED)
        add_library(MNN_Vulkan SHARED IMPORTED)
        add_library(mnncore SHARED IMPORTED)
        set_target_properties(
                MNN
                PROPERTIES IMPORTED_LOCATION
                ${CMAKE_CURRENT_LIST_DIR}/../libs/android/mnn/${ANDROID_ABI}/libMNN.so)

        set_target_properties(
                MNN_CL
                PROPERTIES IMPORTED_LOCATION
                ${CMAKE_CURRENT_LIST_DIR}/../libs/android/mnn/${ANDROID_ABI}/libMNN_CL.so)

        set_target_properties(
                MNN_Express
                PROPERTIES IMPORTED_LOCATION
                ${CMAKE_CURRENT_LIST_DIR}/../libs/android/mnn/${ANDROID_ABI}/libMNN_Express.so)

        set_target_properties(
                MNN_Vulkan
                PROPERTIES IMPORTED_LOCATION
                ${CMAKE_CURRENT_LIST_DIR}/../libs/android/mnn/${ANDROID_ABI}/libMNN_Vulkan.so)

        set_target_properties(
                mnncore
                PROPERTIES IMPORTED_LOCATION
                ${CMAKE_CURRENT_LIST_DIR}/../libs/android/mnn/${ANDROID_ABI}/libmnncore.so)


        list(APPEND LINK_LIBS MNN MNN_CL MNN_Express MNN_Vulkan mnncore)
    else()
        list(APPEND LINK_LIBS MNN)
    endif()


ENDIF()

IF(ENGINE_ORT)
    add_definitions(-DENABLE_ORT)
    file(GLOB_RECURSE ONNXRUNTIME_SOURCE_FILES ${CMAKE_CURRENT_LIST_DIR}/backend/onnxruntime/*.cpp)
    list(APPEND SOURCE_FILES ${ONNXRUNTIME_SOURCE_FILES})
    if(ANDROID)
        add_library(onnxruntime SHARED IMPORTED)

        set_target_properties(
                onnxruntime
                PROPERTIES IMPORTED_LOCATION
                ${CMAKE_CURRENT_LIST_DIR}/../libs/android/onnxruntime/${ANDROID_ABI}/libonnxruntime.so)
    endif()
    list(APPEND LINK_LIBS onnxruntime)
ENDIF()

IF(ENGINE_NCNN)
    add_definitions(-DENABLE_NCNN)
    file(GLOB_RECURSE NCNN_SOURCE_FILES ${CMAKE_CURRENT_LIST_DIR}/backend/ncnn/*.cpp)
    list(APPEND SOURCE_FILES ${NCNN_SOURCE_FILES})
    if(ANDROID)
        set(ncnn_DIR  ${CMAKE_CURRENT_LIST_DIR}/../libs/android/ncnn/install/${ANDROID_ABI}/lib/cmake/ncnn)
        find_package(ncnn REQUIRED)
    endif()

    list(APPEND LINK_LIBS ncnn)
ENDIF()

IF(ENGINE_TNN)
    add_definitions(-DENABLE_TNN)
    file(GLOB_RECURSE TNN_SOURCE_FILES ${CMAKE_CURRENT_LIST_DIR}/backend/tnn/*.cpp)

    link_directories(${CMAKE_CURRENT_LIST_DIR}/../libs/${PLATFORM}/${CMAKE_HOST_SYSTEM_PROCESSOR}/tnn/)
    list(APPEND SOURCE_FILES ${TNN_SOURCE_FILES})

    if(ANDROID)
        add_library(TNN SHARED IMPORTED)

        set_target_properties(
                TNN
                PROPERTIES IMPORTED_LOCATION
                ${CMAKE_CURRENT_LIST_DIR}/../libs/android/tnn/${ANDROID_ABI}/libTNN.so)
    endif()
    list(APPEND LINK_LIBS TNN)
ENDIF()


IF(ENGINE_OPV)
    add_definitions(-DENABLE_OPV)
    include_directories(${CMAKE_CURRENT_LIST_DIR}/../3rdparty/openvino)
    include_directories(${CMAKE_CURRENT_LIST_DIR}/../3rdparty/openvino/ie)
    include_directories(${CMAKE_CURRENT_LIST_DIR}/../3rdparty/openvino/ngraph)

    file(GLOB_RECURSE OPV_SOURCE_FILES ${CMAKE_CURRENT_LIST_DIR}/backend/openvino/*.cpp)
    list(APPEND SOURCE_FILES ${OPV_SOURCE_FILES})
    if(ANDROID)
        add_library(openvino SHARED IMPORTED)

        set_target_properties(
                openvino
                PROPERTIES IMPORTED_LOCATION
                ${CMAKE_CURRENT_LIST_DIR}/../libs/android/openvino/libopenvino.so)
    else()
        link_directories(${CMAKE_CURRENT_LIST_DIR}/../libs/${PLATFORM}/${CMAKE_HOST_SYSTEM_PROCESSOR}/openvino/)
    endif()

    list(APPEND LINK_LIBS openvino)

ENDIF()

IF(ENGINE_PPLite)
    add_definitions(-DENABLE_PPLite)
    include_directories(${CMAKE_CURRENT_LIST_DIR}/../3rdparty/PaddleLite)
    include_directories(${CMAKE_CURRENT_LIST_DIR}/../3rdparty/mklml/include)
    link_directories(${CMAKE_CURRENT_LIST_DIR}/../libs/${PLATFORM}/${CMAKE_HOST_SYSTEM_PROCESSOR}/paddlelite/)
    file(GLOB_RECURSE PPLite_SOURCE_FILES ${CMAKE_CURRENT_LIST_DIR}/backend/paddle_lite/*.cpp)
    list(APPEND SOURCE_FILES ${PPLite_SOURCE_FILES})
    if(ANDROID)
        add_library(paddle_light_api_shared SHARED IMPORTED)

        set_target_properties(
                paddle_light_api_shared
                PROPERTIES IMPORTED_LOCATION
                ${CMAKE_CURRENT_LIST_DIR}/../libs/android/paddle-lite/${ANDROID_ABI}/libpaddle_light_api_shared.so)
        list(APPEND LINK_LIBS paddle_light_api_shared)
    else()
        if(NOT CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "x86_64")
            list(APPEND LINK_LIBS paddle_light_api_shared)
        else()
            list(APPEND LINK_LIBS paddle_full_api_shared iomp5 dl)
        endif()
    endif()

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

file(GLOB_RECURSE PPS_SOURCE_FILES ${CMAKE_CURRENT_LIST_DIR}/preprocess/*.cpp)
file(GLOB_RECURSE UTL_SOURCE_FILES ${CMAKE_CURRENT_LIST_DIR}/utility/*.cpp)
list(APPEND SOURCE_FILES ${PPS_SOURCE_FILES} ${UTL_SOURCE_FILES})

if(NOT ENGINE_NCNN_WASM)
    if(ANDROID)
        add_definitions(-DSHOWLOG)
        find_library( # Sets the name of the path variable.
                log-lib

                # Specifies the name of the NDK library that
                # you want CMake to locate.
                log )
        list(APPEND LINK_LIBS ${log-lib})
#        set(OpenCV_DIR ${PROJECT_SOURCE_DIR}/../../../../../opencv)
        set(OpenCV_DIR ${CMAKE_CURRENT_LIST_DIR}/../libs/android/opencv)

        include_directories(${OpenCV_DIR}/native/jni/include)
        add_library(lib_opencv STATIC IMPORTED)
        set_target_properties(lib_opencv PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_LIST_DIR}/../libs/android/opencv/native/libs/${ANDROID_ABI}/libopencv_java4.so)

        add_library(libc++_shared STATIC IMPORTED)
        set_target_properties(libc++_shared PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_LIST_DIR}/../libs/android/opencv/native/libs/${ANDROID_ABI}/libc++_shared.so)

        list(APPEND SOURCE_FILES aidb_jni.cpp)
        add_library(AiDB SHARED ${SOURCE_FILES})
        target_link_libraries(AiDB libc++_shared lib_opencv ${LINK_LIBS} android -ljnigraphics)

    else()
        set(LIB_PATH ${CMAKE_CURRENT_LIST_DIR}/../libs/${PLATFORM}/${CMAKE_HOST_SYSTEM_PROCESSOR})
        link_directories(${LIB_PATH})
        find_package(OpenCV REQUIRED)
        include_directories(${OpenCV_INCLUDE_DIRS})
        add_library(AiDB SHARED ${SOURCE_FILES})
        target_link_libraries(AiDB ${OpenCV_LIBS} ${LINK_LIBS})
        if(C_API)
#            find_package(Protobuf REQUIRED)
#            include_directories(${Protobuf_INCLUDE_DIRS})
#            include_directories(proto)
#            add_library(AiDB_C SHARED ${SOURCE_FILES} c_api/aidb_c_api.cpp)
#            target_link_libraries(AiDB_C ${OpenCV_LIBS} ${LINK_LIBS} ${Protobuf_LIBRARIES})
            file(GLOB_RECURSE WRAPPER_FILES ${CMAKE_CURRENT_LIST_DIR}/server/wrapper/*.cpp)
            file(GLOB_RECURSE SERVER_FILES ${CMAKE_CURRENT_LIST_DIR}/server/*.cpp ${CMAKE_CURRENT_LIST_DIR}/../3rdparty/cpp-base64/base64.cpp)

            add_library(AiDB_C SHARED ${SOURCE_FILES} ${WRAPPER_FILES} ${SERVER_FILES} c_api/aidb_c_api.cpp)
            target_link_libraries(AiDB_C ${OpenCV_LIBS} ${LINK_LIBS})
        endif()
    endif()


else ()
    add_definitions(-DENABLE_NCNN_WASM)
    set(ncnn_DIR "${CMAKE_CURRENT_LIST_DIR}/../libs/ncnn-20230223-webassembly/${WASM_FEATURE}/lib/cmake/ncnn")
    message(${ncnn_DIR} )
    find_package(ncnn REQUIRED)

#    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -s FORCE_FILESYSTEM=1 -s INITIAL_MEMORY=256MB -s EXIT_RUNTIME=1")
#    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -s FORCE_FILESYSTEM=1 -s INITIAL_MEMORY=256MB -s EXIT_RUNTIME=1")
#    set(CMAKE_EXECUTBLE_LINKER_FLAGS "${CMAKE_EXECUTBLE_LINKER_FLAGS} -s FORCE_FILESYSTEM=1 -s INITIAL_MEMORY=256MB -s EXIT_RUNTIME=1")
#
#    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -sEXPORTED_FUNCTIONS=['_yolov5_ncnn','_malloc','_free'] --preload-file ${CMAKE_CURRENT_SOURCE_DIR}/assets@.")
#    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -sEXPORTED_FUNCTIONS=['_yolov5_ncnn','_malloc','_free'] --preload-file ${CMAKE_CURRENT_SOURCE_DIR}/assets@.")
#    set(CMAKE_EXECUTBLE_LINKER_FLAGS "${CMAKE_EXECUTBLE_LINKER_FLAGS} -sEXPORTED_FUNCTIONS=['_yolov5_ncnn','_malloc','_free'] --preload-file ${CMAKE_CURRENT_SOURCE_DIR}/assets@.")

    if(${WASM_FEATURE} MATCHES "threads")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp -pthread -s USE_PTHREADS=1 -s PTHREAD_POOL_SIZE=4")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -pthread -s USE_PTHREADS=1 -s PTHREAD_POOL_SIZE=4")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fopenmp -pthread -s USE_PTHREADS=1 -s PTHREAD_POOL_SIZE=4")
    endif()

    file(GLOB_RECURSE NCNN_SOURCE_FILES ${CMAKE_CURRENT_LIST_DIR}/backend/ncnn/*.cpp)
    list(APPEND SOURCE_FILES ${NCNN_SOURCE_FILES})

    add_library(AiDB SHARED ${SOURCE_FILES})
    target_link_libraries(AiDB ncnn)
endif ()

#set(INCLUDE_FILES
#        ${CMAKE_CURRENT_LIST_DIR}/core/AIDBData.h
#        ${CMAKE_CURRENT_LIST_DIR}/core/AIDBInput.h
#        ${CMAKE_CURRENT_LIST_DIR}/core/Interpreter.h
#        ${CMAKE_CURRENT_LIST_DIR}/core/StatusCode.h
#        ${CMAKE_CURRENT_LIST_DIR}/utility/Utility.h)
#
#install(FILES ${INCLUDE_FILES}
#        DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../include)




