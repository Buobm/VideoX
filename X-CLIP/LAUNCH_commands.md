To launch 


python -m torch.distributed.launch --nproc_per_node=1 main.py \
-cfg configs/k400/32_8.yaml --output /home/mischa/ETH/semester_project/VideoX/X-CLIP/output --only_test --resume checkpoints/k400_32_8.pth \
--opts TEST.NUM_CLIP 4 TEST.NUM_CROP 3


python -m torch.distributed.launch --nproc_per_node=1 main.py \
-cfg configs/epic_kitchen_100/32_8.yaml --output /home/mischa/ETH/semester_project/VideoX/X-CLIP/output --only_test --resume checkpoints/k400_32_8.pth

python -m torch.distributed.launch --nproc_per_node=1 main.py \
-cfg configs/HoloAssist/32_8.yaml --output /home/mischa/ETH/semester_project/VideoX/X-CLIP/output --only_test