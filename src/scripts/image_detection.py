#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image as PILImage, ImageDraw
import numpy as np
import os
import sys

# Add DEIM module path to sys.path
deim_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '/root/ws/DEIM'))
sys.path.append(deim_path)
from engine.core import YAMLConfig


class DEIMNode(Node):
    def __init__(self, args):
        super().__init__('deim_node')
        self.bridge = CvBridge()
        self.args = args

        # Initialize DEIM model
        self.device = torch.device(self.args.device)
        self.model = self.load_deim_model()

        # Subscribe to the ROS 2 image topic
        self.subscription = self.create_subscription(
            Image,
            '/kinect_camera/image_raw',  # Replace with your camera topic
            self.image_callback,
            10)
        self.get_logger().info("DEIM node started. Waiting for images...")

    def load_deim_model(self):
        """Load the DEIM model."""
        cfg = YAMLConfig(self.args.config, resume=self.args.resume)

        if 'HGNetv2' in cfg.yaml_cfg:
            cfg.yaml_cfg['HGNetv2']['pretrained'] = False

        if self.args.resume:
            checkpoint = torch.load(self.args.resume, map_location='cpu')
            if 'ema' in checkpoint:
                state = checkpoint['ema']['module']
            else:
                state = checkpoint['model']
        else:
            raise AttributeError('Only support resume to load model.state_dict by now.')

        # Load train mode state and convert to deploy mode
        cfg.model.load_state_dict(state)

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = cfg.model.deploy()
                self.postprocessor = cfg.postprocessor.deploy()

            def forward(self, images, orig_target_sizes):
                outputs = self.model(images)
                outputs = self.postprocessor(outputs, orig_target_sizes)
                return outputs

        return Model().to(self.device)

    def image_callback(self, msg):
        """Callback function for processing ROS 2 image messages."""
        try:
            # Convert ROS 2 Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")
            return

        # Convert OpenCV image to PIL image
        pil_image = PILImage.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))

        # Preprocess the image
        orig_size = torch.tensor([[pil_image.width, pil_image.height]]).to(self.device)
        transforms = T.Compose([
            T.Resize((640, 640)),
            T.ToTensor(),
        ])
        im_data = transforms(pil_image).unsqueeze(0).to(self.device)

        # Pass the image through the DEIM model
        with torch.no_grad():
            output = self.model(im_data, orig_size)
        labels, boxes, scores = output

        # Draw detections on the image
        self.draw_detections(pil_image, labels, boxes, scores)

        # Convert PIL image back to OpenCV image for visualization
        cv_image_with_detections = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        # Display the image with detections
        cv2.imshow("DEIM Object Detection", cv_image_with_detections)
        cv2.waitKey(1)

    def draw_detections(self, image, labels, boxes, scores, thrh=0.4):
        """Draw bounding boxes and labels on the image."""
        draw = ImageDraw.Draw(image)

        for i, scr in enumerate(scores):
            lab = labels[i][scr > thrh]
            box = boxes[i][scr > thrh]
            scrs = scr[scr > thrh]

            for j, b in enumerate(box):
                draw.rectangle(list(b), outline='red')
                draw.text((b[0], b[1]), text=f"{lab[j].item()} {round(scrs[j].item(), 2)}", fill='blue')

        image.save('deim_results.jpg')


def main(args=None):
    # Parse command-line arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True, help='Path to DEIM config file')
    parser.add_argument('-r', '--resume', type=str, required=True, help='Path to DEIM model checkpoint')
    parser.add_argument('-d', '--device', type=str, default='cpu', help='Device to run the model on (e.g., "cuda" or "cpu")')
    args = parser.parse_args()

    # Initialize ROS 2
    rclpy.init(args=args)
    deim_node = DEIMNode(args)

    try:
        rclpy.spin(deim_node)
    except KeyboardInterrupt:
        pass
    finally:
        deim_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()