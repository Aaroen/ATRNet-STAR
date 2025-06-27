# 1. 使用官方 NVIDIA CUDA 基础镜像
FROM nvidia/cuda:12.1.1-base-ubuntu22.04

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive

# 2. 安装核心系统工具
RUN apt-get update && apt-get install -y \
    git \
    git-lfs \
    wget \
    nano \
    tmux \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 3. 初始化 Git LFS
RUN git lfs install --system

# 4. 配置 tmux
RUN echo "set -g mouse on" > /root/.tmux.conf

# 5. 安装 Miniconda
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh

# 6. 设置 Conda 的环境变量
ENV PATH /opt/conda/bin:$PATH

# 7. 安装 Mamba 加速器
RUN conda install -y mamba -n base -c conda-forge

# 8. 使用 Mamba 创建环境并安装所有依赖
RUN . /opt/conda/etc/profile.d/conda.sh && conda activate base && \
    echo "==> 使用 Mamba 安装依赖..." && \
    mamba create -n ATRBench -y \
    python=3.11 \
    pytorch torchvision torchaudio pytorch-cuda=12.1 \
    numpy scipy pillow opencv matplotlib tqdm \
    captum timm \
    -c pytorch -c conda-forge -c nvidia

# 9. 正确初始化 conda 和 mamba
RUN /opt/conda/bin/conda init bash && \
    /opt/conda/bin/mamba shell init --shell bash

# 10. 配置自动激活环境
RUN echo 'eval "$(mamba shell hook --shell bash)"' >> /root/.bashrc && \
    echo "mamba activate ATRBench" >> /root/.bashrc

# 11. 配置 Git LFS 全局设置（可选）
RUN git config --global filter.lfs.clean 'git-lfs clean -- %f' && \
    git config --global filter.lfs.smudge 'git-lfs smudge -- %f' && \
    git config --global filter.lfs.process 'git-lfs filter-process' && \
    git config --global filter.lfs.required true

# 12. 设置默认命令
CMD ["/bin/bash"]