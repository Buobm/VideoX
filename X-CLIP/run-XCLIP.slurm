#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --gpus-per-node=1
#SBATCH --gres=gpumem:30g
#SBATCH --time=00:20:00
#SBATCH --mem-per-cpu=8000
#SBATCH --tmp=4000 # per node!!
#SBATCH --job-name=X-CLIP
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
module load gcc/6.3.0 python/3.7.4 cuda/11.3.1 cudnn/8.8.1.3 nccl/2.11.4-1
module load eth_proxy

source myenv/bin/activate

echo "GPUs allocated by SLURM: ${CUDA_VISIBLE_DEVICES}"

FREE_PORT=$(python -c 'import socket; sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM); sock.bind(("", 0)); print(sock.getsockname()[1]); sock.close()')
echo "Free Port: ${FREE_PORT}"
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
python -m torch.distributed.launch --nproc_per_node=4 --master_port=$FREE_PORT main.py  \
-cfg configs/k400/32_8.yaml \
--output /cluster/scratch/buobm/X-CLIP_Output/k400 \
# --only_test \
# --resume /cluster/scratch/buobm/X-CLIP_Output/HoloAssist/best.pth