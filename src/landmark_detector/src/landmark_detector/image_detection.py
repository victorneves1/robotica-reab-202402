import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import torch
import torchvision.transforms as T
import numpy as np
from PIL import Image as PILImage, ImageDraw
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '/root/ws/src')))
from deim.engine.core import YAMLConfig
from landmark_detector_interfaces.msg import Landmark, LandmarkArray


class ROS2LandmarkDetector(Node):
    def __init__(self, model, device):
        super().__init__('landmark_detector')
        
        # Subscribe to image topic
        self.subscription = self.create_subscription(
            Image, '/kinect_camera/image_raw', self.image_callback, 10)
        
        # Publisher for detected landmarks
        self.landmark_publisher = self.create_publisher(LandmarkArray, '/landmarks', 10)

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

        # Publish detected landmarks
        self.publish_landmarks(labels, boxes, scores)
        
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


    def publish_landmarks(self, labels, boxes, scores, thrh=0.4):
        """ Publishes detected objects as ROS2 messages """
        landmark_array_msg = LandmarkArray()

        scrs = scores[0]
        labs = labels[0][scrs > thrh]
        boxs = boxes[0][scrs > thrh]
        scrs = scrs[scrs > thrh]

        for i, b in enumerate(boxs):
            landmark_msg = Landmark()
            landmark_msg.label = str(labs[i].item())
            landmark_msg.bbox = [float(b[0]), float(b[1]), float(b[2]), float(b[3])]
            landmark_msg.confidence = float(scrs[i].item())

            landmark_array_msg.landmarks.append(landmark_msg)

        self.landmark_publisher.publish(landmark_array_msg)
        self.get_logger().info(f'Published {len(landmark_array_msg.landmarks)} landmarks.')


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
    
    # Start ROS2 node
    print("Starting ROS2 node...")
    node = ROS2LandmarkDetector(model, device)
    rclpy.spin(node)

    # Clean up
    print("Shutting down...")
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
