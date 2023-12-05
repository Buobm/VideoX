#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --gpus-per-node=1

#SBATCH --time=00:20:00
#SBATCH --mem-per-cpu=4000
#SBATCH --tmp=4000 # per node!!
#SBATCH --job-name=X-CLIP_32_8_12_view
#SBATCH --output=X-CLIP_32_8_12_view.out
#SBATCH --error=X-CLIP_32_8_12_view.err
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

# python -m torch.distributed.launch --nproc_per_node=1 --master_port=$FREE_PORT main.py  \
# -cfg configs/HoloAssist/14_8.yaml \
# --output /cluster/scratch/buobm/X-CLIP_Output/HoloAssist/enhanced_labels \
# #Train X-CLIP

python -m torch.distributed.launch --nproc_per_node=1 --master_port=$FREE_PORT main.py  \
-cfg configs/HoloAssist/32_8.yaml \
--output /cluster/scratch/buobm/X-CLIP_Output/HoloAssist_pretrained/ViT-B_32_8/HoloAssist_finetune_12_view_test_only/ \
--resume /cluster/scratch/buobm/X-CLIP_Output/HoloAssist_pretrained/ViT-B_32_8/HoloAssist_finetune_train_and_test/ckpt_epoch_19.pth \
--only_test \
--opts TEST.NUM_CLIP 4 TEST.NUM_CROP 3
#--resume /cluster/scratch/buobm/X-CLIP_Output/HoloAssist_pretrained/ViT-B_32_8/HoloAssist_finetune_train_and_test/ckpt_epoch_19.pth \
#--only_test
#--opts TEST.NUM_CLIP 4 TEST.NUM_CROP 3
#--resume checkpoints/Acc@1_49_Acc@5_76_epoch_29.pth \
#ref epic kitchen 1.3 top1 acc