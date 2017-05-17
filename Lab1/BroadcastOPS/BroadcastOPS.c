//
// Created by Andrew on 27.04.2017.
//
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char **argv) {
    if (argc < 2) {
        printf("usage: broadcastOps <number-of-sendings>");
        return 1;
    }
    int numberOfSendings;
    if (sscanf(argv[1], "%i", &numberOfSendings) != 1) {
        fprintf(stderr, "error - not an integer");
        return 1;
    }
    MPI_Init(&argc, &argv);

    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the rank of the process
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    MPI_Status status;
    clock_t start, end;

    double timeForNSendings, timeForNBroadcasts;
    if (world_size < 2) {
        printf("World is too small\n");
        return 1;
    }

    int isMaster = rank == 0;
    if (isMaster) {
        printf("World size is %d\n", world_size);
        printf("Number of tries is %d\n", numberOfSendings);

        start = clock();

        for (int sending = 0; sending < numberOfSendings; ++sending) {
            for (int destination = 1; destination < world_size; ++destination) {
                MPI_Send(NULL, 0, MPI_BYTE, destination, 1, MPI_COMM_WORLD);
            }
        }
        end = clock();
        timeForNSendings = ((double) (end - start) / 1000000.0F) * 1000;
        printf("time for %d send operations = %f \n", numberOfSendings, timeForNSendings);
        printf("OPS = %f\n", (double) numberOfSendings / timeForNSendings);

    } else {
        for (int sending = 0; sending < numberOfSendings; ++sending) {
            MPI_Recv(NULL, 0, MPI_BYTE, 0, 1, MPI_COMM_WORLD, &status);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (isMaster) {
        start = clock();
    }
    for (int sending = 0; sending < numberOfSendings; ++sending) {
        MPI_Bcast(NULL, 0, MPI_BYTE, 0, MPI_COMM_WORLD);
    }
    if (isMaster) {
        end = clock();
        timeForNBroadcasts = ((double) (end - start) / 1000000.0F) * 1000;
        printf("time for %d broadcast operations = %f \n", numberOfSendings, timeForNBroadcasts);
        printf("OPS = %f\n", (double) numberOfSendings / timeForNBroadcasts);
    }

    MPI_Finalize();
}