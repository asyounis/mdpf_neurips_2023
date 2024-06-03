#!/bin/bash
#SBATCH --ntasks 1               							# Number of tasks to run
#SBATCH --nodes 1                							# Ensure that all cores are on one machine
#SBATCH --cpus-per-task=24         							# CPU cores/threads
#SBATCH --gres=gpu:1              							# Number of GPUs (per node)
#SBATCH --mem 24000         								# Reserve 24 GB RAM for the job
#SBATCH --time 2-00:00          							# Max Runtime in D-HH:MM
#SBATCH --partition liv.p   								# Partition to submit to
#SBATCH --job-name bearings_Normal_dc								# The name of the job that is running
#SBATCH --output /scratch/ali/slurm/stdout/bearings_Normal_dc.out  	# File to which STDOUT will be written, %j inserts jobid
#SBATCH --error /scratch/ali/slurm/stdout/bearings_Normal_dc.err  	# File to which STDERR will be written, %j inserts jobid
#SBATCH --nodelist dizzy

# Activate the anaconda session
source ~/anaconda3/etc/profile.d/conda.sh
conda activate ali_particle_vo

# Go to the correct directory
cd /scratch/ali/Development/particle_vo/src/experiments/bearings_only_experiments_methodical_25_particles_multiple_runs/discrete_concrete

# Run!
./run.bash
