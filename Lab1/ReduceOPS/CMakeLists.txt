project(MPI_ReduceOPS)

set(SOURCE_FILES ReduceOPS.c)

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