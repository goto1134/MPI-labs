#!/bin/bash
mpicc /home/andrey.efanov/Lab1/ReduceOPS/ReduceOPS.c -o /home/andrey.efanov/Lab1/ReduceOPS/reduce.out
for i in `seq 8 16`;
    do
        qsub -l nodes=$i:ppn=1 /home/andrey.efanov/Lab1/ReduceOPS/reduce.sh
        #Предположительно за 60 секунд задача выполнится
        sleep $((20*$i))
    done