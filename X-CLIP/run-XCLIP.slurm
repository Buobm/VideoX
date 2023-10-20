#!/bin/bash

#TODO list:
#   variable for number of GPUs
#   get working directory
#   make it easy to switch from train to test


#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --gpus-per-node=2
#SBATCH --time=5:00:00
#SBATCH --mem-per-cpu=4000
#SBATCH --tmp=4000 # per node!!
#SBATCH --job-name=X-CLIP
#SBATCH --output=X-CLIP.out
#SBATCH --error=X-CLIP.err


# Load the required modules (this is an example, adjust based on your environment and needs)
module load gcc/6.3.0
module load python_gpu/3.7.4 #also just loads "cuda" and "cudnn"
module load eth_proxy

source env/bin/activate

echo "GPUs allocated by SLURM: ${CUDA_VISIBLE_DEVICES}"

#launch test run of X-CLIP
python -m torch.distributed.launch --nproc_per_node=2 main.py \
-cfg configs/k400/32_8.yaml \
--output /cluster/home/buobm/Semester_project/VideoX/X-CLIP/output \
--only_test \
--resume checkpoints/k400_32_8.pth \
--opts TEST.NUM_CLIP 4 TEST.NUM_CROP 3