cmake_minimum_required(VERSION 3.7)
project(MPI_ReduceOPS)

find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})

set(CMAKE_C_STANDARD 99)

set(SOURCE_FILES RediceOPS.c)

add_executable(MPI_ReduceOPS ${SOURCE_FILES})
target_link_libraries(MPI_ReduceOPS ${MPI_LIBRARIES})

if (MPI_COMPILE_FLAGS)
    set_target_properties(MPI_ReduceOPS PROPERTIES
            COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
endif ()

if (MPI_LINK_FLAGS)
    set_target_properties(MPI_ReduceOPS PROPERTIES
            LINK_FLAGS "${MPI_LINK_FLAGS}")
endif ()