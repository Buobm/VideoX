#!/bin/bash

#SBATCH -n 1
#SBATCH --time=8:00
#SBATCH --mem-per-cpu=2000
#SBATCH --tmp=4000                        # per node!!
#SBATCH --job-name=X-CLIP
#SBATCH --output=X-CLIP.out
#SBATCH --error=X-CLIP.err


# Load the required modules (this is an example, adjust based on your environment and needs)
module load gcc/6.3.0
module load python_gpu/3.7.4 #also just loads "cuda" and "cudnn"

source X-CLIP_env/bin/activate

#launch test run of X-CLIP
python test.py