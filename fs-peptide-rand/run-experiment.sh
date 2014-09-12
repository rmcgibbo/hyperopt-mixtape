#!/bin/zsh
#PBS -S /bin/zsh
#PBS -l nodes=1:ppn=16
#PBS -o /dev/null
#PBS -e /dev/null
#PBS -l walltime=5:00:00
#PBS -V
#PBS -q MP
cd $PBS_O_WORKDIR

make startserver >| server.log 2>&1 &
sleep 10
mpirun -np 15 make runworker >| worker.log 2>&1
wait
