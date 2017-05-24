//
// Created by Andrew on 23.04.2017.
//
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char **argv) {
    if (argc < 2) {
        printf("usage: MPI_ChannelCapacity <number-of-sendings> <number-of-bytes>");
        return 1;
    }
    int numberOfSendings, numberOfBytes;
    if (sscanf(argv[1], "%i", &numberOfSendings) != 1) {
        fprintf(stderr, "error - not an integer");
        return 1;
    }
    if (sscanf(argv[2], "%i", &numberOfBytes) != 1) {
        fprintf(stderr, "error - not an integer");
        return 1;
    }

    char *bytes = malloc(numberOfBytes * sizeof(char));
    MPI_Init(&argc, &argv);

    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the rank of the process
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    MPI_Status status;
    if (world_size > 1) {
        if (rank == 0) {
            printf("World size is %d\n", world_size);
            printf("Number of tries is %d\n", numberOfSendings);
            int destination;
            for (destination = 1; destination < world_size; ++destination) {
                printf("Counting channel capacity with processor %d of %d\n", destination, world_size - 1);
                double timeForNSendings = 0;

                clock_t start, end;
                start = clock();
                int sending;
                for (sending = 0; sending < numberOfSendings; ++sending) {
                    MPI_Send(bytes, numberOfBytes, MPI_BYTE, destination, 1, MPI_COMM_WORLD);
                    MPI_Recv(bytes, numberOfBytes, MPI_BYTE, destination, 1, MPI_COMM_WORLD, &status);
                }
                end = clock();
                timeForNSendings = (double) (end - start) / CLOCKS_PER_SEC;
                printf("time for %d cycles with %d = %f seconds \n", numberOfSendings, destination, timeForNSendings);
                printf("Channel capacity = %f b/s\n",
                       ((double) numberOfBytes / timeForNSendings * 2 * numberOfSendings));
            }
        } else {
            int sending;
            for (sending = 0; sending < numberOfSendings; ++sending) {
                MPI_Recv(bytes, numberOfBytes, MPI_BYTE, 0, 1, MPI_COMM_WORLD, &status);
                MPI_Send(bytes, numberOfBytes, MPI_BYTE, 0, 1, MPI_COMM_WORLD);
            }
        }
    } else {
        printf("World is too small\n");
        return 1;
    }

    MPI_Finalize();
}