project(aidb_wasm_samples)

add_definitions(-DENABLE_NCNN_WASM)

include_directories(${CMAKE_CURRENT_LIST_DIR}/../3rdparty/)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../source/core)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../source/utility)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../source/)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -sASSERTIONS=1 -sFORCE_FILESYSTEM=1 -sINITIAL_MEMORY=256MB -sEXIT_RUNTIME=1")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -sASSERTIONS=1 -sFORCE_FILESYSTEM=1 -sINITIAL_MEMORY=256MB -sEXIT_RUNTIME=1")
set(CMAKE_EXECUTBLE_LINKER_FLAGS "${CMAKE_EXECUTBLE_LINKER_FLAGS} -sASSERTIONS=1 -sFORCE_FILESYSTEM=1 -sINITIAL_MEMORY=256MB -sEXIT_RUNTIME=1")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -sEXPORTED_FUNCTIONS=['_scrfd_ncnn','_malloc','_free']")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -sEXPORTED_FUNCTIONS=['_scrfd_ncnn','_malloc','_free']")
set(CMAKE_EXECUTBLE_LINKER_FLAGS "${CMAKE_EXECUTBLE_LINKER_FLAGS} -sEXPORTED_FUNCTIONS=['_scrfd_ncnn','_malloc','_free']")

if(${WASM_FEATURE} MATCHES "threads")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp -pthread -s USE_PTHREADS=1 -s PTHREAD_POOL_SIZE=4")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -pthread -s USE_PTHREADS=1 -s PTHREAD_POOL_SIZE=4")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fopenmp -pthread -s USE_PTHREADS=1 -s PTHREAD_POOL_SIZE=4")
endif()

add_executable(aidb-wasm-${WASM_FEATURE} aidb_samples.cpp)
target_link_libraries(aidb-wasm-${WASM_FEATURE} aidb)