FROM nvidia/cuda:12.6.0-base-ubuntu22.04

# Avoid prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# xvfb virtual monitor, x11 utils for ubuntu window handling engine
RUN apt-get update && apt-get install -y \
    x11-utils \
    python3-opengl \
    xvfb \ 
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*


# 2. Python dependencies
RUN pip install --no-cache-dir \
    pyvirtualdisplay \
    torchvision \
    "av<14" \
    vmas \
    benchmarl \
    wandb
 
WORKDIR /workspace

# Start in bash for interactive development
CMD ["/bin/bash"]
