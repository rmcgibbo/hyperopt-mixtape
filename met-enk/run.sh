#!/bin/bash
#PBS -S /bin/bash
#PBS -l nodes=1:ppn=16
#PBS -l walltime=24:00:00
#PBS -V
#PBS -q MP
cd $PBS_O_WORKDIR


make startserver &
sleep 10
mpirun -np 5 make runworker
