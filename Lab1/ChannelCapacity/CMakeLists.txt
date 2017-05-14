cmake_minimum_required(VERSION 3.7)
project(MPI_ChannelCapacity)

find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})

set(CMAKE_C_STANDARD 99)

set(SOURCE_FILES ChannelCapacity.c)


add_executable(MPI_ChannelCapacity ${SOURCE_FILES} ChannelCapacity.c)
target_link_libraries(MPI_ChannelCapacity ${MPI_LIBRARIES})

if(MPI_COMPILE_FLAGS)
	set_target_properties(MPI_ChannelCapacity PROPERTIES
			COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
endif()

if(MPI_LINK_FLAGS)
	set_target_properties(MPI_ChannelCapacity PROPERTIES
			LINK_FLAGS "${MPI_LINK_FLAGS}")
endif()