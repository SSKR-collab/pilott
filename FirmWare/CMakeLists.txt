cmake_minimum_required(VERSION 3.20)
project(LayerWisePilotSim C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Add compile definitions
add_definitions(-D_GNU_SOURCE)

# Platform-specific configurations  
if(WIN32)
    add_definitions(-DWIN32_LEAN_AND_MEAN)
    set(PLATFORM_LIBS pthread m)
else()
    set(PLATFORM_LIBS pthread m)
endif()

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Source files
set(SOURCES
    src/main.c
    src/devices/head_feeder.c
    src/devices/worker_conv1.c
    src/devices/tail_classifier.c
    src/nn/conv1d.c
    src/nn/pooling.c
    src/nn/fully_connected.c
    src/nn/activations.c
    src/nn/optimizers.c
    src/comm/ipc_tensor.c
    src/comm/shm_protocol.c
    src/data/ucr_loader.c
    src/data/tensor.c
    src/threading/worker_threads.c
    src/config/config_loader.c
    src/utils/logging.c
    src/shared_memory.c
)

# Create executable
add_executable(device ${SOURCES})

# Link libraries
target_link_libraries(device ${PLATFORM_LIBS})

# Compiler flags
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(device PRIVATE -g -O0 -Wall -Wextra)
else()
    target_compile_options(device PRIVATE -O3 -DNDEBUG)
endif()

# Copy test data (commented out - using actual UCR dataset instead)
# configure_file(${CMAKE_CURRENT_SOURCE_DIR}/test_data/Cricket_X_TRAIN.txt 
#                ${CMAKE_CURRENT_BINARY_DIR}/Cricket_X_TRAIN.txt COPYONLY)
# configure_file(${CMAKE_CURRENT_SOURCE_DIR}/test_data/Cricket_X_TEST.txt 
#                ${CMAKE_CURRENT_BINARY_DIR}/Cricket_X_TEST.txt COPYONLY)