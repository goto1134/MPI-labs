#!/bin/bash
mpicc /home/andrey.efanov/Lab1/ChannelCapacity/ChannelCapacity.c -o /home/andrey.efanov/Lab1/ChannelCapacity/capacity.out

qsub -l nodes=2:ppn=1 /home/andrey.efanov/Lab1/ChannelCapacity/capacity.sh
sleep 40
qsub -l nodes=1:ppn=2 /home/andrey.efanov/Lab1/ChannelCapacity/capacity.sh
