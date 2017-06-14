cmake_minimum_required(VERSION 3.7)

set(CMAKE_C_STANDARD 90)

find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})

project(MPI_labs)
add_subdirectory(Lab1)
add_subdirectory(Lab2)
