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
    ros-humble-apriltag-msgs \  
    python3-rosdep \
    python3-pip \
    tmux \
    python3-opencv \
    ros-humble-cv-bridge \
    ros-humble-vision-opencv \
    libcanberra-gtk3-module \ 
    x11-apps \
    mesa-utils \
    && rm -rf /var/lib/apt/lists/*

# Install colcon-common-extensions and OpenCV
RUN pip3 install colcon-common-extensions opencv-python

# Install PyTorch with correct CUDA version
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install YOLOv11 dependencies
RUN pip3 install seaborn pandas requests

# Initialize rosdep
RUN if [ ! -f /etc/ros/rosdep/sources.list.d/20-default.list ]; then rosdep init; fi && rosdep update

# Set up workspace
WORKDIR /root/ws
RUN mkdir -p /root/ws/src

# Clone RTAB-Map repositories
RUN git clone https://github.com/introlab/rtabmap.git src/rtabmap
RUN git clone --branch ros2 https://github.com/introlab/rtabmap_ros.git src/rtabmap_ros

# Install dependencies
RUN apt update && rosdep update && rosdep install --from-paths src --ignore-src -r -y

# Set CMAKE_PREFIX_PATH explicitly
RUN echo "export CMAKE_PREFIX_PATH=/root/ws/install:$CMAKE_PREFIX_PATH" >> ~/.bashrc

# Build workspace
RUN /bin/bash -c "source /opt/ros/humble/setup.bash && \
    export MAKEFLAGS='-j3' && \
    colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release -DWITH_GRIDMAP=OFF"

# Clone DEIM and install dependencies
RUN git clone https://github.com/ShihuaHuang95/DEIM.git && pip3 install -r /root/ws/DEIM/requirements.txt

# Install NumPy (fix possible PyTorch compatibility issues)
RUN pip3 install "numpy==1.24.4"

# Ensure ROS2 setup is sourced in all new shells
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc
RUN echo "source /root/ws/install/setup.bash" >> /root/.bashrc

# Ensure rtabmap_ros launch files are installed
RUN mkdir -p /root/ws/install/rtabmap_ros/share/rtabmap_ros/launch
RUN cp -r /root/ws/src/rtabmap_ros/rtabmap_launch/launch/* /root/ws/install/rtabmap_ros/share/rtabmap_ros/launch/

# Run container in interactive mode
CMD ["/bin/bash"]
