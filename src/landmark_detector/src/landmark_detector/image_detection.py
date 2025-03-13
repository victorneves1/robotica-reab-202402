import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import torch
import torchvision.transforms as T
import numpy as np
from PIL import Image as PILImage, ImageDraw
import os
import sys

# Ensure the path to DEIM is available
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '/root/ws/src')))
from deim.engine.core import YAMLConfig
from landmark_detector_interfaces.msg import Landmark, LandmarkArray

class ROS2LandmarkDetector(Node):
    def __init__(self, model, device):
        super().__init__('landmark_detector')
        
        # Subscribe to the RGB image topic
        self.create_subscription(
            Image, '/kinect_camera/image_raw', self.image_callback, 10)
        
        # Subscribe to the camera info topic
        self.create_subscription(
            CameraInfo, '/kinect_camera/camera_info', self.camera_info_callback, 10)
        
        # Subscribe to the depth image topic
        self.create_subscription(
            Image, '/kinect_camera/depth/image_raw', self.depth_callback, 10)
        
        # Publisher for detected landmarks
        self.landmark_publisher = self.create_publisher(LandmarkArray, '/landmarks', 10)

        self.bridge = CvBridge()
        self.device = device
        self.model = model
        self.transform = T.Compose([
            T.Resize((640, 640)),
            T.ToTensor(),
        ])
        
        # Initialize variables for camera info and depth image
        self.camera_width = None
        self.camera_height = None
        self.camera_matrix = None  # Expected to be a list of 9 floats
        self.depth_image = None

    def camera_info_callback(self, msg):
        self.get_logger().info('Receiving camera info...')
        self.camera_width = msg.width
        self.camera_height = msg.height
        self.camera_matrix = msg.k  # row-major 3x3 matrix: [fx, 0, cx, 0, fy, cy, 0, 0, 1]

    def depth_callback(self, msg):
        # Convert depth image message to a CV image using "passthrough"
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except Exception as e:
            self.get_logger().error(f"Depth callback conversion failed: {e}")

    def image_callback(self, msg):
        self.get_logger().info('Receiving RGB image...')
        
        # Convert RGB image message to OpenCV format
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        
        # Convert OpenCV image to PIL for processing
        pil_image = PILImage.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        
        w, h = pil_image.size
        orig_size = torch.tensor([[w, h]]).to(self.device)
        
        # Preprocess image for the model
        im_data = self.transform(pil_image).unsqueeze(0).to(self.device)
        
        # Run inference with your detection model
        labels, boxes, scores = self.model(im_data, orig_size)

        # Publish detected landmarks (now as 3D bounding boxes)
        self.publish_landmarks(labels, boxes, scores)
        
        # Draw 2D bounding boxes on the image for debugging
        self.draw(pil_image, labels, boxes, scores)
        
        # Convert back to OpenCV format for display
        output_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        cv2.imshow("DEIM Output", output_image)
        cv2.waitKey(1)

    def draw(self, image, labels, boxes, scores, thrh=0.4):
        draw = ImageDraw.Draw(image)
        scrs = scores[0]
        labs = labels[0][scrs > thrh]
        boxs = boxes[0][scrs > thrh]
        scrs = scrs[scrs > thrh]

        for i, b in enumerate(boxs):
            draw.rectangle(list(b), outline='red')
            draw.text((b[0], b[1]), text=f"{labs[i].item()} {round(scrs[i].item(), 2)}", fill='blue')

    def publish_landmarks(self, labels, boxes, scores, thrh=0.4):
        """Publishes detected objects as 3D bounding boxes in LandmarkArray."""
        if self.depth_image is None or self.camera_matrix is None:
            self.get_logger().warn("Depth image or camera info not received yet!")
            return

        landmark_array_msg = LandmarkArray()

        scrs = scores[0]
        labs = labels[0][scrs > thrh]
        boxs = boxes[0][scrs > thrh]
        scrs = scrs[scrs > thrh]

        for i, b in enumerate(boxs):
            # Compute 3D bounding box from the 2D bounding box and the depth image
            x_min, y_min, z_min, x_max, y_max, z_max = self.compute_3d_box_from_2d(b, self.depth_image)
            
            landmark_msg = Landmark()
            landmark_msg.label = str(labs[i].item())
            landmark_msg.confidence = float(scrs[i].item())
            landmark_msg.x_min = x_min
            landmark_msg.y_min = y_min
            landmark_msg.z_min = z_min
            landmark_msg.x_max = x_max
            landmark_msg.y_max = y_max
            landmark_msg.z_max = z_max

            landmark_array_msg.landmarks.append(landmark_msg)

        self.landmark_publisher.publish(landmark_array_msg)
        self.get_logger().info(f'Published {len(landmark_array_msg.landmarks)} landmarks.')

    def compute_3d_box_from_2d(self, box2d, depth_image):
        """
        Convert a 2D bounding box into an axis-aligned 3D bounding box using depth data.
        :param box2d: a tensor or list of [x_min, y_min, x_max, y_max]
        :param depth_image: the latest depth image as a numpy array
        :return: (x_min, y_min, z_min, x_max, y_max, z_max)
        """
        # Convert box2d to integers (pixel coordinates)
        x_min_2d = int(box2d[0].item())
        y_min_2d = int(box2d[1].item())
        x_max_2d = int(box2d[2].item())
        y_max_2d = int(box2d[3].item())
        
        # Get the valid image dimensions
        img_height, img_width = depth_image.shape

        # Clamp the coordinates to be within the image bounds
        x_min_2d = max(0, min(x_min_2d, img_width - 1))
        x_max_2d = max(0, min(x_max_2d, img_width - 1))
        y_min_2d = max(0, min(y_min_2d, img_height - 1))
        y_max_2d = max(0, min(y_max_2d, img_height - 1))

        # Define corners: top-left, top-right, bottom-left, bottom-right
        corners = [
            (x_min_2d, y_min_2d),
            (x_max_2d, y_min_2d),
            (x_min_2d, y_max_2d),
            (x_max_2d, y_max_2d)
        ]
        
        # Extract camera intrinsics from self.camera_matrix
        fx = self.camera_matrix[0]
        fy = self.camera_matrix[4]
        cx = self.camera_matrix[2]
        cy = self.camera_matrix[5]

        points_3d = []
        for (u, v) in corners:
            # Ensure indices are within bounds:
            u = max(0, min(u, img_width - 1))
            v = max(0, min(v, img_height - 1))
            depth_val = float(depth_image[v, u])
            if depth_val <= 0.0:
                continue  # Skip invalid depth values
            # Project pixel (u, v) into 3D using the pinhole camera model
            X = (u - cx) * depth_val / fx
            Y = (v - cy) * depth_val / fy
            Z = depth_val
            points_3d.append((X, Y, Z))
        
        if not points_3d:
            # Fallback: return zeros if no valid depth was found
            return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        
        # Compute the axis-aligned 3D bounding box from the valid corner points
        xs, ys, zs = zip(*points_3d)
        return (min(xs), min(ys), min(zs), max(xs), max(ys), max(zs))


def main():
    print("Starting DEIM Image Detection Node...")
    rclpy.init()
    
    # Load model configuration and weights
    config_path = "/root/ws/src/deim/configs/deim_dfine/deim_hgnetv2_s_coco.yml"
    resume_path = "/root/ws/src/models/deim_dfine_hgnetv2_s_coco_120e.pth"
    
    print(f"Loading model from {resume_path}...")
    cfg = YAMLConfig(config_path, resume=resume_path)
    
    if 'HGNetv2' in cfg.yaml_cfg:
        cfg.yaml_cfg['HGNetv2']['pretrained'] = False

    print("Deploying model...")
    checkpoint = torch.load(resume_path, map_location='cpu')
    state = checkpoint['ema']['module'] if 'ema' in checkpoint else checkpoint['model']
    cfg.model.load_state_dict(state)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs
    
    print("Model loaded successfully!")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Model().to(device)
    print(f"Running on device: {device}")
    
    print("Starting ROS2 node...")
    node = ROS2LandmarkDetector(model, device)
    rclpy.spin(node)

    print("Shutting down...")
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
