find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

include_directories(${CMAKE_CURRENT_LIST_DIR}/../3rdparty/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../source/core)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../source/utility)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../source/)

link_directories(${CMAKE_CURRENT_LIST_DIR}/../libs/${PLATFORM}/${CMAKE_HOST_SYSTEM_PROCESSOR}/)


IF(ENGINE_TNN)
    link_directories(${CMAKE_CURRENT_LIST_DIR}/../libs/${PLATFORM}/${CMAKE_HOST_SYSTEM_PROCESSOR}/tnn/)
ENDIF()

IF(ENGINE_OPV)
    link_directories(${CMAKE_CURRENT_LIST_DIR}/../libs/${PLATFORM}/${CMAKE_HOST_SYSTEM_PROCESSOR}/openvino/)
ENDIF()

IF(ENGINE_PPLite)
    link_directories(${CMAKE_CURRENT_LIST_DIR}/../libs/${PLATFORM}/${CMAKE_HOST_SYSTEM_PROCESSOR}/paddlelite/)
ENDIF()

IF(ENABLE_SD)
    #  too slow
    #    include(FetchContent)
    #
    #    FetchContent_Declare(
    #            ddim_scheduler_cpp
    #            GIT_REPOSITORY https://github.com/TalkUHulk/ddim_scheduler_cpp.git
    #            GIT_TAG        0.0.1
    #    )
    #    FetchContent_MakeAvailable(ddim_scheduler_cpp)

    #    FetchContent_Declare(
    #            tokenizers-cpp
    #            GIT_REPOSITORY https://github.com/mlc-ai/tokenizers-cpp.git
    #            GIT_TAG        main
    #            GIT_PROGRESS   TRUE
    #    )
    #    FetchContent_MakeAvailable(tokenizers-cpp)

    set(TOKENZIER_CPP_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../3rdparty/tokenizers-cpp)
    add_subdirectory(${TOKENZIER_CPP_PATH} tokenizers)

    set(DDIM_SCHEDULER_CPP_PATH  ${CMAKE_CURRENT_SOURCE_DIR}/../3rdparty/ddim-scheduler-cpp)
    add_subdirectory(${DDIM_SCHEDULER_CPP_PATH} ddim_scheduler)


    list(APPEND EXTRA_LIBS tokenizers_cpp ddim_scheduler_cpp)

ENDIF()

file(GLOB_RECURSE SAMPLES_FILES ${CMAKE_CURRENT_LIST_DIR}/*.cpp)
foreach (filepath IN LISTS SAMPLES_FILES)
    get_filename_component(filename ${filepath} NAME_WE)
    if(${filename} MATCHES "StableDiffusion.*" AND NOT ENABLE_SD)
        continue()
    endif()

    add_executable(${filename} ${filepath})
    target_include_directories(${filename} PRIVATE
            ${TOKENZIER_CPP_PATH}/include
            ${ddim_scheduler_cpp_SOURCE_DIR}/source)
    target_link_libraries(${filename} PRIVATE AiDB ${EXTRA_LIBS} ${OpenCV_LIBS})
endforeach ()