//
// Created by Andrew on 27.04.2017.
//
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char **argv) {
    if (argc < 2) {
        printf("usage: reduceOps <number-of-sendings>");
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

    double timeForNSendings, timeForNReduce;

    if (world_size < 2) {
        printf("World is too small\n");
        return 1;
    }

    if (rank == 0) {
        printf("World size is %d\n", world_size);
        printf("Number of tries is %d\n", numberOfSendings);

        start = clock();

        for (int sending = 0; sending < numberOfSendings; ++sending) {
            for (int destination = 1; destination < world_size; ++destination) {
                MPI_Recv(NULL, 0, MPI_BYTE, destination, 1, MPI_COMM_WORLD, &status);
            }
        }
        end = clock();
        timeForNSendings = ((double) (end - start) / 1000000.0F) * 1000;
        printf("time for %d receive operations = %f \n", numberOfSendings, timeForNSendings);
        printf("OPS = %f\n", timeForNSendings / (double) numberOfSendings);

    } else {
        for (int sending = 0; sending < numberOfSendings; ++sending) {
            MPI_Send(NULL, 0, MPI_BYTE, 0, 1, MPI_COMM_WORLD);
        }
    }

    start = clock();
    for (int sending = 0; sending < numberOfSendings; ++sending) {
        MPI_Reduce(NULL, NULL, 0, MPI_BYTE, 0, 0, MPI_COMM_WORLD);
    }
    end = clock();
    timeForNReduce = ((double) (end - start) / 1000000.0F) * 1000;
    printf("time for %d reduce operations = %f \n", numberOfSendings, timeForNReduce);

    MPI_Finalize();
}