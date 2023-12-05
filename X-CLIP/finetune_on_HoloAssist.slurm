#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --gpus-per-node=2
#SBATCH --gres=gpumem:20g
#SBATCH --time=02:00:00
#SBATCH --mem-per-cpu=4000
#SBATCH --tmp=4000 # per node!!
#SBATCH --job-name=X-CLIP_32_8_finetune
#SBATCH --output=X-CLIP_32_8_finetune.out
#SBATCH --error=X-CLIP_32_8_finetune.err
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

python -m torch.distributed.launch --nproc_per_node=2 --master_port=$FREE_PORT main.py  \
-cfg configs/HoloAssist/32_8_finetune.yaml \
--output /cluster/scratch/buobm/X-CLIP_Output/HoloAssist_pretrained/ViT-B_32_8/HoloAssist_finetune_5_5_split \
--resume checkpoints/ViT-B_32__8_HoloAssist_All_Acc@1_23.4_Acc@5_55.6.pth
#--only_test