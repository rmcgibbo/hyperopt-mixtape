#!/bin/zsh
#PBS -S /bin/zsh
#PBS -l nodes=2:ppn=16
#PBS -o /dev/null
#PBS -e /dev/null
#PBS -l walltime=10:00:00
#PBS -V
#PBS -q MP
cd $PBS_O_WORKDIR
cat $PBS_NODEFILE

make startserver >| server.log 2>&1 &
sleep 10
mpirun -wdir . -f $PBS_NODEFILE -np 35 make runworker >| worker.log 2>&1
wait
