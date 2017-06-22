//
// Created by Andrew on 30.05.2017.
//

#include <mpi.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

typedef struct {
    int mainRow;
    int rowToSwapWithMain;
} MainSwap;

static const int MASTER_NODE_RANK = 0;

static const int INITIAL_MATRIX_TAG = 1;

static const int CURRENT_WORKER_TAG = 2;

int minimumEquationCount(int dataTypeSizeInBytes, int numberOfComputationalNodes, int minimumSizeOfSystemPerNode);

int getEquationCountForDouble(int worldSize);

//void fillMatrix(int numberOfEquations, int worldSize, int index, double *matrix);

void generateColumnBlock(int numberOfEquations, int worldSize, int index, double *block);

void generateResultArray(int equationCount, double *matrix);

double generateNormalizedRandom();

int indexOfMaxInBounds(double *matrix, int lowerBound, int upperBound);

void calculateMultipliers(int mainRow, int mainColumn, int equationCount, double *matrix, double *multipliers);

int indexInLocalMatrix(int equationCount, int row, int column);

void modifyMatrix(int equationCount, int columnsPerNode, int mainRow, double *matrix, double *multipliers);

void swapRows(int equationCount, int columnsPerNode, int firstRow, int secondRow, double *matrix);

int main(int argc, char **argv) {

    MPI_Init(&argc, &argv);

    // Get the number of processes
    int worldSize;
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    // Get the rank of the process
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    MPI_Status status;
    clock_t start, end;
    double timeForNSendings, timeForNReduce;

    if (worldSize < 2) {
        printf("World is too small\n");
        return 1;
    }

    int isMaster = rank == MASTER_NODE_RANK;
    int equationCount;
    int columnsPerNode;
    int oneBlockSize;

    if (isMaster) {
        printf("World size is %d\n", worldSize);

        equationCount = getEquationCountForDouble(worldSize);
        columnsPerNode = equationCount / (worldSize - 1);
        oneBlockSize = equationCount * columnsPerNode;
        printf("Equation count is %d\nwith %d per node\nwith size of %d\n", equationCount, columnsPerNode,
               oneBlockSize);
    }

    MPI_Bcast(&equationCount, 1, MPI_INT, MASTER_NODE_RANK, MPI_COMM_WORLD);
    MPI_Bcast(&columnsPerNode, 1, MPI_INT, MASTER_NODE_RANK, MPI_COMM_WORLD);
    MPI_Bcast(&oneBlockSize, 1, MPI_INT, MASTER_NODE_RANK, MPI_COMM_WORLD);

    double *matrix = malloc(oneBlockSize * sizeof(double));
    if (isMaster) {
        printf("Starting the sending of data\n");
        srand((unsigned int) time(0));
        int node;
        for (node = 1; node < worldSize; ++node) {
            generateColumnBlock(equationCount, worldSize, node - 1, matrix);
            MPI_Send(matrix, oneBlockSize, MPI_DOUBLE, node, INITIAL_MATRIX_TAG, MPI_COMM_WORLD);
        }
        free(matrix);
        matrix = malloc(equationCount * sizeof(double));
        generateResultArray(equationCount, matrix);
    } else {
        MPI_Recv(matrix, oneBlockSize, MPI_DOUBLE, MASTER_NODE_RANK, INITIAL_MATRIX_TAG, MPI_COMM_WORLD, &status);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    //Прямой ход метода Гаусса, приведение к треугольному виду.
    MainSwap pair;
    int activeNode;
    for (activeNode = 1; activeNode < worldSize; ++activeNode) {
        int column;
        for (column = 0; column < columnsPerNode; ++column) {
            if (rank == activeNode) {
                int mainRow = columnsPerNode * (activeNode - 1) + column;
                int mainIndex = indexInLocalMatrix(equationCount, mainRow, column);
                int upperBound = indexInLocalMatrix(equationCount, equationCount, column);
                int maxIndex = indexOfMaxInBounds(matrix, mainIndex, upperBound);
                pair.mainRow = mainRow;
                pair.rowToSwapWithMain = mainRow + maxIndex - mainIndex;
                if (pair.mainRow != pair.rowToSwapWithMain) {
                    printf("Swap %d, %d\n", pair.mainRow, pair.rowToSwapWithMain);
                }
            }
            MPI_Bcast(&pair, 1, MPI_2INT, activeNode, MPI_COMM_WORLD);
            if (pair.mainRow != pair.rowToSwapWithMain) {
                swapRows(equationCount, isMaster ? 1 : columnsPerNode, pair.mainRow, pair.rowToSwapWithMain, matrix);
            }
            //запоминаем главную строку
            int mainRow = pair.mainRow;
            double *multipliers = calloc((size_t) equationCount, sizeof(double));
            if (rank == activeNode) {
                calculateMultipliers(mainRow, column, equationCount, matrix, multipliers);
            }
            MPI_Bcast(multipliers, equationCount, MPI_DOUBLE, activeNode, MPI_COMM_WORLD);
            modifyMatrix(equationCount, isMaster ? 1 : columnsPerNode, mainRow, matrix, multipliers);
            free(multipliers);
        }
    }

    //Матрица треугольная, обратный ход метода Гаусса.

    free(matrix);
    MPI_Finalize();
}

void swapRows(int equationCount, int columnsPerNode, int firstRow, int secondRow, double *matrix) {
    int column;
    double buf;
    int mainRowElementIndex;
    int secondRowElementIndex;
    for (column = 0; column < columnsPerNode; ++column) {
        mainRowElementIndex = indexInLocalMatrix(equationCount, firstRow, column);
        secondRowElementIndex = indexInLocalMatrix(equationCount, secondRow, column);
        buf = matrix[mainRowElementIndex];
        matrix[mainRowElementIndex] = matrix[secondRowElementIndex];
        matrix[secondRowElementIndex] = buf;
    }
}

/**
 * Изменяет матрицу при прямом проходе, вычитая из строк элементы главной строки помноженные на заданные множители
 * @param equationCount
 * @param columnsPerNode
 * @param mainRow
 * @param matrix локальная матрица
 * @param multipliers множители
 */
void modifyMatrix(int equationCount, int columnsPerNode, int mainRow, double *matrix, double *multipliers) {
    int row;
    for (row = mainRow + 1; row < equationCount; ++row) {
        if (multipliers[row] != 0) {
            int column;
            for (column = 0; column < columnsPerNode; ++column) {
                int indexInMatrix = indexInLocalMatrix(equationCount, row, column);
                int mainIndexInColumn = indexInLocalMatrix(equationCount, mainRow, column);
                matrix[indexInMatrix] = matrix[indexInMatrix] - matrix[mainIndexInColumn] * multipliers[row];
            }
        }
    }
}

/**
 * @param equationCount количество уравнений в системе
 * @param row строка
 * @param column столбец
 * @return Индекс элемента в локальной матрице
 */
int indexInLocalMatrix(int equationCount, int row, int column) {
    return row + equationCount * column;
}

/**
 * Вычисляет множители для преобразования матрицы
 * @param mainRow строка главного элемента
 * @param mainColumn колонка с главным элементом
 * @param equationCount
 * @param matrix
 * @param multipliers массив множителей
 */
void calculateMultipliers(int mainRow, int mainColumn, int equationCount, double *matrix, double *multipliers) {
    int mainIndex = indexInLocalMatrix(equationCount, mainRow, mainColumn);
    int row;
    for (row = mainRow + 1; row < equationCount; ++row) {
        multipliers[row] = matrix[indexInLocalMatrix(equationCount, row, mainColumn)] / matrix[mainIndex];
    }
}

/**
 * @param matrix локальная матрица
 * @param lowerBound нижняя граница, включена
 * @param upperBound верхняя грацица, не включена
 * @return Индекс максимального элемента в промежутке
 */
int indexOfMaxInBounds(double *matrix, int lowerBound, int upperBound) {
    double max = matrix[lowerBound];
    int maxIndex = lowerBound;
    int index;
    for (index = lowerBound + 1; index < upperBound; ++index) {
        if (max < matrix[index]) {
            max = matrix[index];
            maxIndex = index;
        }
    }
    return maxIndex;
}

int getEquationCountForDouble(int worldSize) {
    int doubleSize = sizeof(double);
    int oneHundredMegaBytes = 100 * 1024 * 1024;
    int minValue = minimumEquationCount(doubleSize, worldSize - 1, oneHundredMegaBytes);
    return minValue - minValue % (worldSize - 1) + worldSize - 1;
}

void generateColumnBlock(int numberOfEquations, int worldSize, int index, double *block) {
    int columnCount = numberOfEquations / (worldSize - 1);
    int column;
    int row;
    int mainIndex;
    int currentIndex;
    double normalizedRandom;
    for (column = 0; column < columnCount; ++column) {
        mainIndex = indexInLocalMatrix(numberOfEquations, index * columnCount + column, column);
        for (row = 0; row < columnCount; ++row) {
            currentIndex = indexInLocalMatrix(numberOfEquations, row, column);
            normalizedRandom = generateNormalizedRandom();
            block[currentIndex] = currentIndex == mainIndex
                                  ? 10 + normalizedRandom * 100
                                  : normalizedRandom * 10;
        }
    }
}

void generateResultArray(int equationCount, double *matrix) {
    int row;
    for (row = 0; row < equationCount; ++row) {
        matrix[row] = generateNormalizedRandom() * 20 - 10;
    }
}

double generateNormalizedRandom() { return rand() / (double) RAND_MAX; }

//void fillMatrix(int numberOfEquations, int worldSize, int index, double *block) {
//    int blockSize = numberOfEquations / (worldSize - 1);
//    int i;
//    int j;
//    int mainIndex;
//    for (i = 0; i < blockSize; ++i) {
//        for (j = 0; j < numberOfEquations - 1; ++j) {
//            mainIndex = index * (worldSize - 1) + i;
//            block[i * numberOfEquations + j] = j == mainIndex
//                                               ? 10 + (double) rand() / (double) RAND_MAX * 100
//                                               : ((double) rand() / (double) RAND_MAX) * 10;
//        }
//        block[i * numberOfEquations + numberOfEquations - 1] = ((double) rand() / (double) RAND_MAX) * 20 - 10;
//    }
//}

int minimumEquationCount(int dataTypeSizeInBytes, int numberOfComputationalNodes, int minimumSizeOfSystemPerNode) {
    return (int) ceil((sqrt((minimumSizeOfSystemPerNode * numberOfComputationalNodes)
                            / (double) dataTypeSizeInBytes)));
}
