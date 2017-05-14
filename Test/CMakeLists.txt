project(MPI_Lab)

find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})

set(CMAKE_C_STANDARD 99)

set(SOURCE_FILES hello_world_mpi.c)


add_executable(MPI_Latency hello_world_mpi.c Lab1/Latency/Latency.c)
target_link_libraries(MPI_Latency ${MPI_LIBRARIES})

if(MPI_COMPILE_FLAGS)
	set_target_properties(MPI_Latency PROPERTIES
			COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
endif()

if(MPI_LINK_FLAGS)
	set_target_properties(MPI_Latency PROPERTIES
			LINK_FLAGS "${MPI_LINK_FLAGS}")
endif()