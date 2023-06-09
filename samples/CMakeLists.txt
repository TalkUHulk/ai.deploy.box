find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

include_directories(${CMAKE_CURRENT_LIST_DIR}/../3rdparty/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../source/core)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../source/utility)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../source/)

link_directories(${CMAKE_CURRENT_LIST_DIR}/../libs/${PLATFORM}/)


IF(ENGINE_TNN)
    link_directories(${CMAKE_CURRENT_LIST_DIR}/../libs/${PLATFORM}/tnn/)
ENDIF()

IF(ENGINE_OPV)
    link_directories(${CMAKE_CURRENT_LIST_DIR}/../libs/${PLATFORM}/openvino/intel64/)
ENDIF()

IF(ENGINE_PPLite)
    link_directories(${CMAKE_CURRENT_LIST_DIR}/../libs/${PLATFORM}/paddlelite/)
ENDIF()

file(GLOB_RECURSE SAMPLES_FILES ${CMAKE_CURRENT_LIST_DIR}/*.cpp)
foreach (filepath IN LISTS SAMPLES_FILES)
    get_filename_component(filename ${filepath} NAME_WE)
    add_executable(${filename} ${filepath})
    target_link_libraries(${filename} PRIVATE AiDB ${LINK_LIBS} ${OpenCV_LIBS})
endforeach ()