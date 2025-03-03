import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseWithCovarianceStamped
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as T
import numpy as np
from PIL import Image as PILImage, ImageDraw
from cv_bridge import CvBridge
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../deim/')))
from engine.core import YAMLConfig


class ROS2ImageLocalizationSubscriber(Node):
    def __init__(self, model, device):
        super().__init__('image_localization_subscriber')

        # Subscribe to image topic
        self.image_sub = self.create_subscription(
            Image, '/kinect_camera/image_raw', self.image_callback, 10)

        # Subscribe to RTAB-Map localization pose topic
        self.pose_sub = self.create_subscription(
            PoseWithCovarianceStamped, '/rtabmap/localization_pose', self.pose_callback, 10)

        self.bridge = CvBridge()
        self.device = device
        self.model = model
        self.transform = T.Compose([
            T.Resize((640, 640)),
            T.ToTensor(),
        ])

        self.current_pose = None  # Store the latest localization info

    def pose_callback(self, msg):
        """ Callback to receive the robot's estimated pose from RTAB-Map. """
        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation
        self.current_pose = (position.x, position.y, position.z)
        self.get_logger().info(f"RTAB-Map Localization: X={position.x:.2f}, Y={position.y:.2f}, Z={position.z:.2f}")

    def image_callback(self, msg):
        """ Callback to process images and overlay detection results with localization. """
        self.get_logger().info('Receiving image...')

        # Convert ROS2 Image message to OpenCV format
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        # Convert OpenCV image to PIL for processing
        pil_image = PILImage.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))

        w, h = pil_image.size
        orig_size = torch.tensor([[w, h]]).to(self.device)

        # Preprocess image for the model
        im_data = self.transform(pil_image).unsqueeze(0).to(self.device)

        # Run inference
        labels, boxes, scores = self.model(im_data, orig_size)

        # Draw results on the image
        self.draw(pil_image, labels, boxes, scores)

        # Convert back to OpenCV format for display
        output_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        # Overlay localization info if available
        if self.current_pose:
            x, y, z = self.current_pose
            cv2.putText(output_image, f"Pose: X={x:.2f}, Y={y:.2f}, Z={z:.2f}",
                        (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Show the processed image
        cv2.imshow("DEIM + RTAB-Map Localization", output_image)
        cv2.waitKey(1)  # Update the display

    def draw(self, image, labels, boxes, scores, thrh=0.4):
        """ Draw bounding boxes and labels on the image. """
        draw = ImageDraw.Draw(image)
        scrs = scores[0]
        labs = labels[0][scrs > thrh]
        boxs = boxes[0][scrs > thrh]
        scrs = scrs[scrs > thrh]

        for i, b in enumerate(boxs):
            draw.rectangle(list(b), outline='red')
            draw.text((b[0], b[1]), text=f"{labs[i].item()} {round(scrs[i].item(), 2)}", fill='blue')


def main():
    rclpy.init()

    # Load DEIM Model
    config_path = "/root/ws/src/deim/configs/deim_dfine/deim_hgnetv2_s_coco.yml"
    resume_path = "/root/ws/src/models/deim_dfine_hgnetv2_s_coco_120e.pth"

    cfg = YAMLConfig(config_path, resume=resume_path)

    if 'HGNetv2' in cfg.yaml_cfg:
        cfg.yaml_cfg['HGNetv2']['pretrained'] = False

    checkpoint = torch.load(resume_path, map_location='cpu')
    state = checkpoint['ema']['module'] if 'ema' in checkpoint else checkpoint['model']
    cfg.model.load_state_dict(state)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.depl
