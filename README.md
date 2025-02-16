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
xhost +local:docker
docker run -it --rm \
    --net=host \
    --env="DISPLAY" \
    --env="QT_X11_NO_MITSHM=1" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix" \
    --volume="$(pwd):/root/ws" \
    ros2-humble-gazebo-classic
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