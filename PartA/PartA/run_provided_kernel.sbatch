#!/bin/bash

#SBATCH --account=ece_gy_9143-2024fa
##SBATCH --partition=n1s8-v100-1
##SBATCH --partition=g2-standard-12
#SBATCH --partition=c12m85-a100-1
#SBATCH --job-name=modified_kernel
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --output=output/output_%x_%j.txt
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --mem=8G

echo "Problem 1 Q1"
singularity exec --nv\
    --overlay /scratch/zl5604/lab4/lab4.ext3:ro \
    /scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
    /bin/bash -c 'cd /scratch/zl5604/lab4; \
    echo "Make with provided kernel"; \
    make clean; make;'

Ns=(500 1000 2000)
for N in ${Ns[@]}; do
    echo "Run vecadd00 using vecaddKernel00 N=$N"
    singularity exec --nv\
        --overlay /scratch/zl5604/lab4/lab4.ext3:ro \
        /scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
        /bin/bash -c './vecadd00 $N;'
done

sizes=(256 512 1024)
for size in ${sizes[@]}
    echo "Run matmult00 using matmultKernel00 size=$size"
    singularity exec --nv\
        --overlay /scratch/zl5604/lab4/lab4.ext3:ro \
        /scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
        /bin/bash -c './matmult00 $size'
done
