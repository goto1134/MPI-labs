//
// Created by Andrew on 30.05.2017.
//

//
// Created by Andrew on 27.04.2017.
//
#include <mpi.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <stdlib.h>

static const int MASTER_NODE_RANK = 0;

static const int INITIAL_MATRIX_TAG = 1;

static const int CURRENT_WORKER_TAG = 2;

int getEquationCountForDouble(int rank);

double *generateMatrix(int numberOfEquations, int worldSize, int index);

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

    int isMaster = rank == MASTER_NODE_RANK;
    int equationCount;
    int equationsPerNode;
    int oneBlockSize;

    if (isMaster) {
        printf("World size is %d\n", world_size);

        equationCount = getEquationCountForDouble(rank);
        equationsPerNode = equationCount / (world_size - 1);
        oneBlockSize = (equationCount + 1) * equationsPerNode;
    }
    MPI_Bcast(&equationCount, 1, MPI_INT, MASTER_NODE_RANK, MPI_COMM_WORLD);
    MPI_Bcast(&equationsPerNode, 1, MPI_INT, MASTER_NODE_RANK, MPI_COMM_WORLD);
    MPI_Bcast(&oneBlockSize, 1, MPI_INT, MASTER_NODE_RANK, MPI_COMM_WORLD);

    double *matrix;
    if (isMaster) {
        printf("Starting the sending of data");
        srand((unsigned int) time(0));

        for (int i = 1; i < world_size; ++i) {
            matrix = generateMatrix(equationCount, world_size, i - 1);
            MPI_Send(matrix, oneBlockSize, MPI_DOUBLE, i, INITIAL_MATRIX_TAG, MPI_COMM_WORLD);
            free(matrix);
        }
    } else {
        MPI_Recv(matrix, oneBlockSize, MPI_DOUBLE, MASTER_NODE_RANK, INITIAL_MATRIX_TAG, MPI_COMM_WORLD, &status);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Данные разосланы.




//    if (isMaster) {
//        start = clock();
//    }
//
//    char receive = 0;
//    int sending;
//    for (sending = 0; sending < numberOfSendings; ++sending) {
//        MPI_Reduce(&rank, &receive, 1, MPI_BYTE, MPI_SUM, 0, MPI_COMM_WORLD);
//    }
//
//    if (isMaster) {
//        end = clock();
//        timeForNReduce = (double) (end - start) / CLOCKS_PER_SEC;
//        printf("time for %d reduce operations = %f \n", numberOfSendings, timeForNReduce);
//        printf("OPS = %f\n", (double) numberOfSendings / timeForNReduce);
//
//    }
    MPI_Finalize();
}

int getEquationCountForDouble(int rank) {
    double doubleSize = (double) sizeof(double);
    int minValue = (int) ceil(((sqrt(doubleSize + 400. * (rank - 1)) / doubleSize) - 1.) / 2.);
    return minValue + minValue % (rank - 1);
}

double *generateMatrix(int numberOfEquations, int worldSize, int index) {
    int blockSize = numberOfEquations / (worldSize - 1);
    double *block = malloc(sizeof(double) * blockSize * (numberOfEquations + 1));
    for (int i = 0; i < numberOfEquations; ++i) {
        for (int j = 0; j < numberOfEquations; ++j) {
            int mainIndex = index * (worldSize - 1) + i;
            block[i * blockSize + j] = ((double) rand() / (double) RAND_MAX) * (j != mainIndex ? 10. : 100.);
        }
        block[i * numberOfEquations] = ((double) rand() / (double) RAND_MAX) * 20 - 10;
    }
    return block;
}
