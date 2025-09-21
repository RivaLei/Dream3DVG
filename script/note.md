# pytorch==1.13.1

# CUDA 11.6

conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia

# CUDA 11.7

conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia

# CPU Only

conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 cpuonly -c pytorch

# ROCM 5.2 (Linux only)

pip install torch==1.13.1+rocm5.2 torchvision==0.14.1+rocm5.2 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/rocm5.2

# CUDA 11.6

pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116

# CUDA 11.7

pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117

# CPU only

pip install torch==1.13.1+cpu torchvision==0.14.1+cpu torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cpu

riva 复制 nef

nef3.8.131.12.1+cu11611.6

[torch cuda 版本暂时不更改]

unsethttp_proxyhttps_proxy && pipinstalltorch==1.13.1+cu117torchvision==0.14.1+cu117torchaudio==0.13.1--extra-index-urlhttps://download.pytorch.org/whl/cu117

【不更改的话 diff-raster 编译有问题】

conda install python=3.10 -y

【失败】--暂时不管

(3dvg) lz@lz-Z790M-AORUS-ELITE-AX:~/project/Dream3DVG$ conda install python=3.10 -y
Solving environment: failed

UnsatisfiableError: The following specifications were found to be in conflict:

defaults/linux-64::setuptools==75.1.0=py38h06a4308_0 -> python[version='>=3.8,<3.9.0a0'] -> libffi[version='>=3.2.1,<3.3a0']
defaults/linux-64::setuptools==75.1.0=py38h06a4308_0 -> python[version='>=3.8,<3.9.0a0'] -> readline[version='>=7.0,<8.0a0']
python=3.10
Use "conda info `<package>`" to see the dependencies for each package.
看到conda升级Python时出现了依赖冲突。这通常是因为当前环境中的一些包与Python 3.10不兼容。最好的解决方案是创建一个新的Python 3.10环境。



# diffusion

init_StableDiffusion_pipeline()

连接预训练模型与SVG生成：将HuggingFace的Stable Diffusion模型适配为SVG生成任务

需要下载很大的预训练模型--直接home内存不够。。。。

/home/lz/.conda/envs/3dvg/lib/python3.8/site-packages/huggingface_hub/file_download.py:991: UserWarning: Not enough free disk space to download the file. The expected file size is: 3463.73 MB. The target location /home/lz/.cache/huggingface/hub/models--stabilityai--stable-diffusion-2-1-base/blobs only has 2024.44 MB free disk space.

如果实在不能下载，就不初始化diffusion 本来就不想用到文本-sketch

【最近就得买了 只有10+G 双11买内存放到ubuntu】
