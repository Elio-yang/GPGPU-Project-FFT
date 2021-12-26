cmake_minimum_required(VERSION 3.10)
project(FFT)

set(CMAKE_CXX_STANDARD 11)

# if you want to use google benchmark
#link_libraries(pthread benchmark)

add_executable(ordinary OM.cpp)
add_executable(basic rFFT.cpp)
