#!/bin/bash
#SBATCH --nodes 2
#SBATCH --ntasks-per-node 28
#SBATCH --cpus-per-task=1
#SBATCH --time 24:00:00
#SBATCH --job-name xjob
#SBATCH --output small_job-out
#SBATCH --error small_job-err
#SBATCH --mem-per-cpu=4G

export SLURM_STEP_NODELIST=$SLURM_NODELIST
echo $SLURM_PROCID
source /hildafs/projects/phy200017p/share/ana/setupLsstim1.sh
source fpfs_config

cd $PWD
mpirun -n 56 impt_summary_hsc.py --config ./config_procsim.ini --min_id 0 --max_id 4000 --magcut 24.5 --mpi
