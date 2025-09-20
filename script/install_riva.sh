#!/bin/bash

set -e

# # Conda setup and environment creation
# eval "$(conda shell.bash hook)"

# unset http_proxy https_proxy
# conda create --name 3dvg python=3.10 --yes
# conda activate 3dvg
# echo "The conda environment was successfully created"

# # Install PyTorch and related libraries
# # NOTE: use "nvcc -V" to find a cuda version satisfied with your systerm, change the command below following "https://pytorch.org/get-started/previous-versions/"
# conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
unset http_proxy https_proxy && pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117

# echo "Pytorch installation is complete."

# # Install common Python dependencies
unset http_proxy https_proxy && pip install hydra-core omegaconf
unset http_proxy https_proxy && pip install freetype-py shapely svgutils cairosvg plyfile open3d
unset http_proxy https_proxy && pip install opencv-python scikit-image matplotlib visdom wandb BeautifulSoup4
unset http_proxy https_proxy && pip install triton numba
unset http_proxy https_proxy && pip install numpy scipy scikit-fmm einops timm fairscale
unset http_proxy https_proxy && pip install accelerate==0.33.0 transformers huggingface_hub==0.24.2 safetensors datasets
unset http_proxy https_proxy && pip install easydict scikit-learn webdataset
unset http_proxy https_proxy && pip install cssutils open3d
echo "The basic dependency library is installed."

# Additional utility libraries
unset http_proxy https_proxy && pip install ftfy regex tqdm
unset http_proxy https_proxy && pip install git+https://github.com/jonbarron/robust_loss_pytorch
unset http_proxy https_proxy && pip install git+https://github.com/openai/CLIP.git
echo "Additional utility installation is complete."

# Install diffusers
unset http_proxy https_proxy && pip install diffusers==0.20.2
echo "Diffusers installation is complete. version: 0.20.2"

# Install xformers (should match torch version, eg. torch 1.13.1 - xformers 0.0.16)
unset http_proxy https_proxy && pip install xformers==0.0.16

# Clone and set up DiffVG, handling dependencies on Ubuntu
# git clone https://github.com/BachiLi/diffvg.git
cd diffvg
git submodule update --init --recursive


# for Ampere sm_80
# sed -i 's/set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11")/set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11 -gencode=arch=compute_80,code=sm_80")/' CMakeLists.txt

# for Ada Lovelace sm_89 (RTX 4070 Ti SUPER)
sed -i 's/set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11")/set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11 -gencode=arch=compute_89,code=sm_89")/' CMakeLists.txt

# Install system dependencies for Ubuntu (to avoid potential issues)
echo "Installing system dependencies for DiffVG..."
sudo apt update
sudo apt install -y cmake ffmpeg build-essential libjpeg-dev libpng-dev libtiff-dev

unset http_proxy https_proxy && pip install svgwrite svgpathtools cssutils torch-tools

# Install DiffVG
python setup.py install
echo "DiffVG installation is complete."
cd ..

# 3DGS
pip install submodules/diff-gaussian-rasterization/
pip install submodules/simple-knn/

# point-e
unset http_proxy https_proxy && pip install git+https://github.com/openai/point-e.git

# Final confirmation
echo "The running environment has been successfully installed!!!"