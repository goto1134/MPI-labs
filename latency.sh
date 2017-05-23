#!/bin/bash
mpicc /home/andrey.efanov/Lab1/Latency/Latency.c -o /home/andrey.efanov/Lab1/Latency/latency.out

qsub -l nodes=2:ppn=1 /home/andrey.efanov/Lab1/Latency/latency.sh
sleep 120
qsub -l nodes=1:ppn=2 /home/andrey.efanov/Lab1/Latency/latency.sh
