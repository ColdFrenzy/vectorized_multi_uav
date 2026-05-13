FROM nvidia/cuda:12.6.0-base-ubuntu22.04
# build using docker build --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) --build-arg USER_NAME=$(whoami)
ARG USER_ID
ARG GROUP_ID
ARG USER_NAME

# Avoid prompts during installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_DEFAULT_TIMEOUT=1000
# python and gpu rendering related stuff
RUN apt-get update && apt-get install -y \
    python3 python3-pip sudo tmux git\
    libglvnd0 libgl1 libglx0 libegl1 libgles2 \
    libglu1-mesa libgl1-mesa-dev libosmesa6 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Create user matching host UID/GID to share privilege in volume
RUN groupadd -g ${GROUP_ID} ${USER_NAME} && \
    useradd -l -u ${USER_ID} -g ${GROUP_ID} -s /bin/bash -m ${USER_NAME} && \
    adduser ${USER_NAME} sudo && \
    echo "${USER_NAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
USER ${USER_NAME}
WORKDIR /home/${USER_NAME}/workspace
RUN echo "set-option -g default-shell /bin/bash" > /home/${USER_NAME}/.tmux.conf

# Python dependencies
RUN python3 -m pip install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
    
RUN python3 -m pip install --no-cache-dir \
    "av<14" \
    vmas \
    benchmarl \
    wandb
 
CMD ["bin", "bash"]
