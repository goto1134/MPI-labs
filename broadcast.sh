#!/bin/bash
mpicc /home/andrey.efanov/Lab1/BroadcastOPS/BroadcastOPS.c -o /home/andrey.efanov/Lab1/BroadcastOPS/broadcast.out
for i in `seq 2 16`;
    do
        qsub -l nodes=$i:ppn=1 /home/andrey.efanov/Lab1/BroadcastOPS/broadcast.sh
        #Предположительно за 60 секунд задача выполнится
        sleep 60
    done