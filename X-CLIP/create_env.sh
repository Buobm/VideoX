module load gcc/6.3.0 python/3.7.4 cuda/11.3.1 cudnn/8.8.1.3 nccl/2.11.4-1 git

python -m venv myenv

source myenv/bin/activate

pip install --upgrade pip

#install specific cuda version of pytorch
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113

pip install -r requirements.txt
pip install packaging
pip install wheel
pip install urllib3==1.26.5
pip install pandas



git clone https://github.com/NVIDIA/apex

cd apex

pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./

cd ..
