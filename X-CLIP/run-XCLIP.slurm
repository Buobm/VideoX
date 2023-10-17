#!/bin/bash

#SBATCH --job-name=DeepLearningJob     # Job name
#SBATCH --partition=general            # Partition (queue) name
#SBATCH --ntasks=1                     # Run on a single CPU
#SBATCH --mem=16gb                     # Memory total in MB (for all cores)
#SBATCH --gres=gpu:2                   # Request two GPUs
#SBATCH --time=10:00:00                # Total run time limit (HH:MM:SS)
#SBATCH --output=slurm.%N.%j.out       # STDOUT output file
#SBATCH --error=slurm.%N.%j.err        # STDERR output file (optional)

# Load the required modules (this is an example, adjust based on your environment and needs)
module load gcc/6.3.0
module load python_gpu/3.7.4 #also just loads "cuda" and "cudnn"

source x-clip_env/bin/activate

#launch test run of X-CLIP
python -m torch.distributed.launch --nproc_per_node=1 main.py \
-cfg configs/k400/32_8.yaml \
--output /home/mischa/ETH/semester_project/VideoX/X-CLIP/output \
--only_test \
--resume checkpoints/k400_32_8.pth \
--opts TEST.NUM_CLIP 4 TEST.NUM_CROP 3