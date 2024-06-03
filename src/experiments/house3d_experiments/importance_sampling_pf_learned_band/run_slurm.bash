#!/bin/bash
#SBATCH --ntasks 1               							# Number of tasks to run
#SBATCH --nodes 1                							# Ensure that all cores are on one machine
#SBATCH --cpus-per-task=24         							# CPU cores/threads
#SBATCH --gres=gpu:1              							# Number of GPUs (per node)
#SBATCH --mem 64000         								# Reserve 24 GB RAM for the job
#SBATCH --time 5-00:00          							# Max Runtime in D-HH:MM
#SBATCH --partition liv.p   								# Partition to submit to
#SBATCH --job-name house_3d_dis_lb							# The name of the job that is running
#SBATCH --output /scratch/ali/slurm/stdout/house_3d_dis_lb.out 	# File to which STDOUT will be written, %j inserts jobid
#SBATCH --error /scratch/ali/slurm/stderr/house_3d_dis_lb.err  	# File to which STDERR will be written, %j inserts jobid
#SBATCH --nodelist dizzy

# Activate the anaconda session
source ~/anaconda3/etc/profile.d/conda.sh
conda activate ali_particle_vo

# Go to the correct directory
cd /scratch/ali/Development/particle_vo/src/experiments/house3d_experiments/importance_sampling_pf_learned_band

# Make the save directory
mkdir -p /scratch/ali/slurm/combined/house_3d/

# Run!
output_file=/scratch/ali/slurm/combined/house_3d/dis_lb.txt
./run.bash > $output_file 2>&1
