#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --gpus-per-node=4
#SBATCH --time=4:00:00
#SBATCH --mem-per-cpu=8000
#SBATCH --tmp=4000 # per node!!
#SBATCH --job-name=X-CLIP_4GPU
#SBATCH --output=X-CLIP.out
#SBATCH --error=X-CLIP.err
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END

#TODO list:
#   variable for number of GPUs
#   get working directory
#   make it easy to switch from train to test
# check if env exists and meets all requirements from requirements.txt

# Load the required modules
module load gcc/6.3.0
module load python_gpu/3.7.4 #also just loads "cuda" and "cudnn"
module load eth_proxy

source env/bin/activate

echo "GPUs allocated by SLURM: ${CUDA_VISIBLE_DEVICES}"

#tensorboard --logdir output/tonsorboard/ --bind_all --port=9876

#tensorboard --logdir output/tonsorboard/

#launch test run of X-CLIP
# python -m torch.distributed.launch --nproc_per_node=2 main.py \
# -cfg configs/k400/32_8.yaml \
# --output /cluster/home/buobm/Semester_project/VideoX/X-CLIP/output \
# --only_test \
# --resume checkpoints/k400_32_8.pth \
# --opts TEST.NUM_CLIP 4 TEST.NUM_CROP 3

#Train X-CLIP
python -m torch.distributed.launch --nproc_per_node=4 main.py \
-cfg configs/epic_kitchen_100/32_8.yaml \
--output /cluster/scratch/buobm/X-CLIP_Output \
--resume /cluster/scratch/buobm/X-CLIP_Output/best.pth