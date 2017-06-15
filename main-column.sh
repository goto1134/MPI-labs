#!/bin/bash
mpicc /home/andrey.efanov/Lab2/MainElementInColumn/MainElementInColumn.c -o /home/andrey.efanov/Lab2/MainElementInColumn/main-column.out -lm

qsub -l nodes=3:ppn=1 /home/andrey.efanov/Lab2/MainElementInColumn/main-column.sh
