project(MPI_BroadcastOPS)

set(SOURCE_FILES BroadcastOPS.c)

add_executable(MPI_BroadcastOPS ${SOURCE_FILES})
target_link_libraries(MPI_BroadcastOPS ${MPI_LIBRARIES})

if (MPI_COMPILE_FLAGS)
    set_target_properties(MPI_BroadcastOPS PROPERTIES
            COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
endif ()

if (MPI_LINK_FLAGS)
    set_target_properties(MPI_BroadcastOPS PROPERTIES
            LINK_FLAGS "${MPI_LINK_FLAGS}")
endif ()