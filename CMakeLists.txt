cmake_minimum_required(VERSION 3.10)
project(matrix_mult CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(matrix_mult main.cu headers/matrix.h src/mul_kernel.cu headers/mul_kernel.h headers/settings.h src/test.cu headers/test.h)

set_target_properties(
    hw
    PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON)