#!/bin/bash
mpicc /home/andrey.efanov/Lab3/boltzmann.c -o /home/andrey.efanov/Lab3/boltzmann.out -lm -std=c99

qsub -l nodes=$1:ppn=$2 /home/andrey.efanov/Lab3/boltzman.sh
