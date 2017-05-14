cmake_minimum_required(VERSION 3.7)
project(MPI_Latency)

find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})

set(CMAKE_C_STANDARD 99)

set(SOURCE_FILES Latency.c)


add_executable(MPI_Latency ${SOURCE_FILES})
target_link_libraries(MPI_Latency ${MPI_LIBRARIES})

if(MPI_COMPILE_FLAGS)
	set_target_properties(MPI_Latency PROPERTIES
			COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
endif()

if(MPI_LINK_FLAGS)
	set_target_properties(MPI_Latency PROPERTIES
			LINK_FLAGS "${MPI_LINK_FLAGS}")
endif()