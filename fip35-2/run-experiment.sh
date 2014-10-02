#!/bin/zsh
#PBS -S /bin/zsh
#PBS -l nodes=2:ppn=15
#PBS -o /dev/null
#PBS -e /dev/null
#PBS -l walltime=48:00:00
#PBS -V
#PBS -q MP
cd $PBS_O_WORKDIR
cat $PBS_NODEFILE

module load mvapich2-psm-x86_64
make startserver >| server.log 2>&1 &
sleep 10
/usr/lib64/mvapich2-psm/bin/mpirun -wdir . -f $PBS_NODEFILE -np 15 -binding rr make runworker >| worker.log 2>&1
wait
