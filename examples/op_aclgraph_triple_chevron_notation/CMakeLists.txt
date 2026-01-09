cmake_minimum_required(VERSION 3.16)
find_package(ASC REQUIRED)
project(kernel_samples LANGUAGES ASC CXX)

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
message("Python3_INCLUDE_DIRS is ${Python3_INCLUDE_DIRS}")

execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_CMAKE_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
find_package(Torch REQUIRED HINTS ${TORCH_CMAKE_PATH}/Torch)

execute_process(
    COMMAND python3 -c "import os, torch_npu; print(os.path.dirname(torch_npu.__file__))"
    OUTPUT_VARIABLE TORCH_NPU_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
set(TORCH_NPU_INCLUDE_DIRS ${TORCH_NPU_PATH}/include)
set(TORCH_NPU_LIBRARIES ${TORCH_NPU_PATH}/lib)

add_library(custom_ops SHARED
    add_custom.asc
    trig_inplace_custom.asc
)

target_include_directories(custom_ops PRIVATE
    ${Python3_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
    ${TORCH_NPU_INCLUDE_DIRS}
)

target_link_libraries(custom_ops PRIVATE
    torch_npu
)

target_link_directories(custom_ops PRIVATE
    ${TORCH_NPU_LIBRARIES}
)

target_compile_options(custom_ops PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-2201>
)
