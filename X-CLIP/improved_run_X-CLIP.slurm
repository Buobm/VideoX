#!/bin/bash


#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --gpus-per-node=1
#SBATCH --time=3:00:00
#SBATCH --mem-per-cpu=4000
#SBATCH --tmp=4000 # per node!!
#SBATCH --job-name=$JOB_NAME
#SBATCH --output=${JOB_NAME}.out
#SBATCH --error=${JOB_NAME}.err

# Load the required modules
module load gcc/6.3.0
module load python_gpu/3.7.4


# Variables
MODE="train"  # Default mode is train. Change to "test" for testing.
JOB_NAME="X-CLIP"
CONFIG="configs/k400/32_8.yaml"
CHECKPOINT="checkpoints/k400_32_8.pth"


source env/bin/activate

echo "GPUs allocated by SLURM: ${CUDA_VISIBLE_DEVICES}"

# Command flags based on MODE
if [ "$MODE" == "test" ]; then
  CMD_FLAGS="--only_test --resume $CHECKPOINT --opts TEST.NUM_CLIP 4 TEST.NUM_CROP 3"
else
  CMD_FLAGS="" # Add training-specific flags here if needed
fi

# Launch the desired mode of X-CLIP
python -m torch.distributed.launch --nproc_per_node=2 main.py \
-cfg $CONFIG \
--output /cluster/home/buobm/Semester_project/VideoX/X-CLIP/output \
$CMD_FLAGS