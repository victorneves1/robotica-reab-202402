# Base image with ROS2 Humble Desktop
FROM osrf/ros:humble-desktop

# Install CUDA dependencies
RUN apt-get update && apt-get install -y \
    software-properties-common \
    wget curl \
    && wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb \
    && dpkg -i cuda-keyring_1.0-1_all.deb \
    && apt-get update && apt-get install -y \
    cuda-toolkit-12-2 \
    && rm -rf /var/lib/apt/lists/*

# Install NVIDIA Container Toolkit (for GPU passthrough)
RUN apt-get update && apt-get install -y nvidia-container-toolkit

# Install necessary ROS2 dependencies
RUN apt-get update && apt-get install -y \
    ros-humble-gazebo-ros-pkgs \
    ros-humble-bcr-bot \
    ros-humble-rviz2 \
    ros-humble-xacro \
    ros-humble-teleop-twist-keyboard \
    ros-humble-navigation2 \
    ros-humble-robot-localization \
    ros-humble-nav2-bringup \
    python3-rosdep \
    python3-pip \
    tmux \
    python3-opencv \
    ros-humble-cv-bridge \
    ros-humble-vision-opencv \
    && rm -rf /var/lib/apt/lists/*

# Install colcon-common-extensions and OpenCV
RUN pip3 install colcon-common-extensions opencv-python

# Install PyTorch with correct CUDA version
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install YOLOv11 dependencies
RUN pip3 install seaborn pandas requests

# Initialize rosdep (only if not already initialized)
RUN if [ ! -f /etc/ros/rosdep/sources.list.d/20-default.list ]; then rosdep init; fi && rosdep update

# Set up workspace
WORKDIR /root/ws

# Clone DEIM and install dependencies
RUN git clone https://github.com/ShihuaHuang95/DEIM.git && pip3 install -r /root/ws/DEIM/requirements.txt

# Build workspace
RUN /bin/bash -c "source /opt/ros/humble/setup.bash && colcon build"

# Install NumPy (fix possible PyTorch compatibility issues)
RUN pip3 install "numpy==1.24.4"

# Set default environment
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc

# Run container in interactive mode
CMD ["/bin/bash"]
