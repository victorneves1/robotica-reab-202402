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
    # ros-humble-rtabmap-ros \
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
# RUN apt update && apt install -y \
    # ros-humble-grid-map-core
#     python3-colcon-common-extensions \
#     ros-humble-rosidl-default-generators \
#     ros-humble-rosidl-default-runtime \
#     ros-humble-rosidl-cmake \
#     ros-humble-rosidl-typesupport-c \
#     ros-humble-rosidl-typesupport-cpp \
#     ros-humble-rosidl-typesupport-introspection-c \
#     ros-humble-rosidl-typesupport-introspection-cpp

RUN mkdir -p /root/ws/src
RUN git clone https://github.com/introlab/rtabmap.git src/rtabmap
RUN git clone --branch ros2 https://github.com/introlab/rtabmap_ros.git src/rtabmap_ros
RUN apt update && rosdep update && rosdep install --from-paths src --ignore-src -r -y
# Build workspace. If you have less than 16 GB of RAM, you may want to reduce the number of jobs (-j6) to avoid running out of memory.
RUN /bin/bash -c "source /opt/ros/humble/setup.bash && export MAKEFLAGS='-j3' && colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release -DWITH_GRIDMAP=OFF"
# RUN colcon build --symlink-install

# Clone DEIM and install dependencies
RUN git clone https://github.com/ShihuaHuang95/DEIM.git && pip3 install -r /root/ws/DEIM/requirements.txt

# Install NumPy (fix possible PyTorch compatibility issues)
RUN pip3 install "numpy==1.24.4"

# Set default environment
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc
RUN echo "source /root/ws/install/setup.bash" >> /root/.bashrc

RUN cp /root/ws/src/rtabmap_ros/rtabmap_launch/launch/*.launch.py /root/ws/install/rtabmap_ros/share/rtabmap_ros/

# Run container in interactive mode
CMD ["/bin/bash"]
