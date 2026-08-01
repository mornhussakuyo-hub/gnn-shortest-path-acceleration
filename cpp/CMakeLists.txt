cmake_minimum_required(VERSION 3.16)
project(aic_cpp_online_benchmark LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(aic_cpp_online_benchmark aic_cpp_online_benchmark.cpp)
target_compile_options(aic_cpp_online_benchmark PRIVATE
    $<$<CONFIG:Release>:-O3;-DNDEBUG;-march=native>
    -Wall -Wextra -Wpedantic
)
