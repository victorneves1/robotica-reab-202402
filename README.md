# 🤖 Robótica para Reabilitação — 2024/02
Repositório do projeto de robótica para reabilitação, com simulação, controle e detecção de marcos em ambientes indoor.

---

## 📦 Clonando o Repositório

Clone o repositório com os submódulos:

```bash
git clone --recurse-submodules git@github.com:victorneves1/robotica-reab-202402.git
```

Atualize os submódulos:

```bash
git submodule update --remote --recursive
```

---

## 🐳 Rodando com Docker

### 🔧 Construir a imagem Docker:

```bash
docker build -t ros2-humble-gazebo-classic .
```

### ▶️ Executar container temporário:

```bash
xhost +local:docker &&
docker run -it --rm \
    --net=host \
    --env="DISPLAY" \
    --env="QT_X11_NO_MITSHM=1" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix" \
    --volume="$(pwd):/root/ws/src/landmark_detector" \
    --volume="$(pwd)/rosbag/:/rosbag" \
    --gpus all \
    ros2-humble-gazebo-classic
```

### 📌 Criar container persistente:

```bash
xhost +local:docker &&
docker run -it --name landmark_detector --net=host \
    --env="DISPLAY" \
    --env="QT_X11_NO_MITSHM=1" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix" \
    --volume="$(pwd)/src:/root/ws/src" \
    --volume="$(pwd)/rosbag/:/rosbag" \
    --gpus all \
    ros2-humble-gazebo-classic
```

### 🚀 Iniciar o container persistente:

```bash
xhost +local:docker && docker start -ai landmark_detector
```

### 🧹 Limpar container:

```bash
docker stop landmark_detector && docker rm landmark_detector
```

---

## 🧠 Interagindo com o Robô

### 🖥️ Usando o tmux:

```bash
tmux
```

**Atalhos úteis do tmux:**

```text
- Dividir painel horizontal: Ctrl+B, depois %
- Dividir vertical: Ctrl+B, depois "
- Alternar painéis: Ctrl+B, depois seta
- Nova janela: Ctrl+B, depois C
- Sair do tmux: Ctrl+B, depois D
- Reentrar: tmux attach
```

---

### 🕹️ Simulação e Controle

#### Inicializar simulação:

```bash
ros2 launch bcr_bot gazebo.launch.py
```

#### Ver tópicos ROS:

```bash
ros2 topic list
```

#### Exibir dados dos sensores (exemplos):

```bash
ros2 topic echo /scan
ros2 topic echo /kinect_camera/image_raw
```

#### Mover o robô:

**Avançar:**

```bash
ros2 topic pub /cmd_vel geometry_msgs/msg/Twist \
"{linear: {x: 1.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}" -1
```

**Parar:**

```bash
ros2 topic pub /cmd_vel geometry_msgs/msg/Twist \
"{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}" -1
```

Ou use o script mais amigável:

```bash
python3 scripts/robot_controller.py
```

---

## 🗺️ Rodando o RTAB-Map

```bash
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

---

## 🧠 Detecção de Marcos (Landmarks)

### 1. Compilar o pacote de interfaces:

```bash
colcon build --packages-select landmark_detector_interfaces --symlink-install
source install/setup.bash
```

### 2. Compilar o detector:

```bash
colcon build --packages-select landmark_detector --symlink-install
source install/setup.bash
```

### 3. Rodar os nós:

```bash
ros2 run landmark_detector image_detection
ros2 run landmark_detector landmark_marker_publisher
```

### 4. Rodar o controlador do robô:

```bash
ros2 run landmark_detector robot_controller
```

### 5. Verificar detecções:

```bash
ros2 topic echo /markerarray
```

---

## 📂 Rodando arquivos rosbag

(precisa descomentar algumas coisas nos scripts python)

### 1. Rodar o rosbag:

```bash
source /opt/ros/humble/setup.bash
source /root/ws/install/setup.bash
cd /rosbag
ros2 bag play subset_0.db3 --loop
```

### 2. Publicar parâmetros da câmera:

```bash
python3 src/landmark_detector/src/landmark_detector/publish_camera_parameters.py
```

### 3. Iniciar RViz2:

```bash
ros2 run rviz2 rviz2
```

Adicione manualmente a câmera RGB e imagem de profundidade.

### 4. Rodar o detector:

```bash
ros2 run landmark_detector image_detection
```

---

## 🛠️ Troubleshooting

### RTAB-Map sem exibir mapa?

```bash
rm /root/.ros/rtabmap.db
```

### Problemas ao rodar DEIM?

- Crie um arquivo vazio chamado `__init__.py` dentro da pasta `deim`.

---
