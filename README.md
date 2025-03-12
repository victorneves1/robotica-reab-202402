# robotica-reab-202402
Projeto de Robótica para Reabilitação 2024/02

## How to clone the repo

### Clone it
`git clone --recurse-submodules git@github.com:victorneves1/robotica-reab-202402.git`

### Update submodules
`git submodule update --remote --recursive`

## How to run

### Build the docker image
`docker build -t ros2-humble-gazebo-classic .`


### Run the docker container (in the root of the project)
```
xhost +local:docker &&
docker run -it --rm \
    --net=host \
    --env="DISPLAY" \
    --env="QT_X11_NO_MITSHM=1" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix" \
    --volume="$(pwd):/root/ws/src/landmark_detector" \
    --gpus all \
    ros2-humble-gazebo-classic
```

Alternatively, create a persistent container:
```
xhost +local:docker &&
docker run -it --name landmark_detector --net=host \
    --env="DISPLAY" \
    --env="QT_X11_NO_MITSHM=1" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix" \
    --volume="$(pwd)/src:/root/ws/src" \
    --gpus all \
    ros2-humble-gazebo-classic
```

Then later you can start the container with:
```
xhost +local:docker && docker start -ai landmark_detector
```
If you ever need to clean up:
```
docker stop landmark_detector
    docker rm landmark_detector
```

## Interacting with the robot

### Start tmux session
`tmux`

How to use tmux:
```
Split the Terminal Pane:

    To split horizontally: Press Ctrl+B, then %.
    To split vertically: Press Ctrl+B, then ".

Switch Between Panes or Windows:

    Use Ctrl+B, then arrow keys to navigate panes.
    To create a new window: Press Ctrl+B, then C.

Detach and Reattach Tmux Session:

    Detach from tmux: Press Ctrl+B, then D.
    Reattach to tmux: Run tmux attach.
```
### Launch the simulation

Inside a tmux window run:

`ros2 launch bcr_bot gazebo.launch.py`

### List all active ROS topics:

In another tmux window run:

`ros2 topic list`

Inspect live sensor data (e.g., /scan for LiDAR data or /camera/image_raw for the camera):

`ros2 topic echo /scan`
`ros2 topic echo /kinect_camera/image_raw`

### Move the Robot

Publish a command to make the robot move forward:

`ros2 topic pub /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 1.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}" -1
`

Stop the robot:

`ros2 topic pub /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}" -1
`

You can also use the script in "scripts/robot_controller.py" to control the robot in a user friendly way.

To run the rtabmap:

```
ros2 launch rtabmap_ros rtabmap.launch.py \
 use_sim_time:=true \
 approx_sync:=true \
 approx_sync_max_interval:=0.02 \
 rgb_topic:=/kinect_camera/image_raw \
 depth_topic:=/kinect_camera/depth/image_raw \
 camera_info_topic:=/kinect_camera/camera_info \
 odom_topic:=/odom \
 topic_queue_size:=50 \
 sync_queue_size:=50
```

To run object detection:

Build the interfaces package first:

```
colcon build --packages-select landmark_detector_interfaces --symlink-install
source install/setup.bash
```

Then build the Python package:

```
colcon build --packages-select landmark_detector --symlink-install
source install/setup.bash
```

Run your node:

```
ros2 run landmark_detector image_detection
```

To run the robot controller:

```
ros2 run landmark_detector robot_controller
```

To check the published detections:

```
ros2 topic echo /landmarks
```