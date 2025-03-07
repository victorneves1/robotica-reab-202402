import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as T
import numpy as np
from PIL import Image as PILImage, ImageDraw
from cv_bridge import CvBridge
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../deim/')))
from engine.core import YAMLConfig


class ROS2ImageSubscriber(Node):
    def __init__(self, model, device):
        super().__init__('image_subscriber')
        
        self.subscription = self.create_subscription(
            Image, '/kinect_camera/image_raw', self.image_callback, 10)
        
        self.bridge = CvBridge()
        self.device = device
        self.model = model
        self.transform = T.Compose([
            T.Resize((640, 640)),
            T.ToTensor(),
        ])

    def image_callback(self, msg):
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
        
        # Show the processed image
        cv2.imshow("DEIM Output", output_image)
        cv2.waitKey(1)  # Update the display

    def draw(self, image, labels, boxes, scores, thrh=0.4):
        draw = ImageDraw.Draw(image)
        scrs = scores[0]
        labs = labels[0][scrs > thrh]
        boxs = boxes[0][scrs > thrh]
        scrs = scrs[scrs > thrh]

        for i, b in enumerate(boxs):
            draw.rectangle(list(b), outline='red')
            draw.text((b[0], b[1]), text=f"{labs[i].item()} {round(scrs[i].item(), 2)}", fill='blue')


def main():
    print("Starting DEIM Image Detection Node...")
    rclpy.init()
    
    # Load model
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

    class Model(nn.Module):
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
    
    # Start ROS2 node
    print("Starting ROS2 node...")
    image_subscriber = ROS2ImageSubscriber(model, device)
    rclpy.spin(image_subscriber)

    # Clean up
    print("Shutting down...")
    image_subscriber.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()