# Base image with ROS2 Humble and Gazebo Classic
FROM osrf/ros:humble-desktop

# Install necessary system dependencies
RUN apt-get update && apt-get install -y \
    ros-humble-gazebo-ros-pkgs \
    ros-humble-bcr-bot \
    python3-rosdep \
    python3-pip \
    tmux \
    python3-opencv \
    ros-humble-cv-bridge \
    ros-humble-vision-opencv \
    && rm -rf /var/lib/apt/lists/*

# Install colcon-common-extensions and PyTorch with CUDA support
RUN pip3 install colcon-common-extensions torch torchvision opencv-python

# Install YOLOv11 dependencies
RUN pip3 install seaborn pandas

# Initialize rosdep (only if not already initialized)
RUN if [ ! -f /etc/ros/rosdep/sources.list.d/20-default.list ]; then rosdep init; fi && rosdep update

# Set up workspace
WORKDIR /root/ws
COPY . /root/ws

# Build workspace
RUN /bin/bash -c "source /opt/ros/humble/setup.bash && colcon build"

# Set default environment
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
CMD ["/bin/bash"]