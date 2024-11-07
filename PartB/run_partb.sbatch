#!/bin/bash

#SBATCH --account=ece_gy_9143-2024fa
##SBATCH --partition=n1s8-v100-1
##SBATCH --partition=g2-standard-12
#SBATCH --partition=c12m85-a100-1
#SBATCH --job-name=part_b
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --output=output/output_%x_%j.txt
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --mem=16G

# Define the Singularity image and overlay
SINGULARITY_IMAGE=/scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif
OVERLAY_IMAGE=/scratch/zl5604/lab4/lab4.ext3

# Run make to compile the programs inside the Singularity container
singularity exec --nv \
    --overlay ${OVERLAY_IMAGE}:ro \
    ${SINGULARITY_IMAGE} \
    /bin/bash -c "make -f Makefile_linux"

# Define the values of K
K_VALUES=(1 5 10 50 100)

# Run q1, q2, q3 with each value of K inside the Singularity container
for K in "${K_VALUES[@]}"; do
    singularity exec --nv \
        --overlay ${OVERLAY_IMAGE}:ro \
        ${SINGULARITY_IMAGE} \
        /bin/bash -c "./q1 $K"
    
    singularity exec --nv \
        --overlay ${OVERLAY_IMAGE}:ro \
        ${SINGULARITY_IMAGE} \
        /bin/bash -c "./q2 $K"
    
    singularity exec --nv \
        --overlay ${OVERLAY_IMAGE}:ro \
        ${SINGULARITY_IMAGE} \
        /bin/bash -c "./q3 $K"
done