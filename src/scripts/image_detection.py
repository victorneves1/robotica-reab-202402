#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import torch
from torchvision import transforms
import numpy as np

class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('image_subscriber')
        self.subscription = self.create_subscription(
            Image,
            '/kinect_camera/image_raw',
            self.image_callback,
            10)
        self.bridge = CvBridge()
        self.get_logger().info("Image subscriber node started.")

        # Load YOLOv11 model
        self.model = self.load_yolov11_model()
        self.classes = self.model.names  # Get class names
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.get_logger().info(f"Using device: {self.device}")

    def load_yolov11_model(self):
        # Load YOLOv11 model (replace with your YOLOv11 model path)
        model = torch.hub.load('ultralytics/yolov5', 'custom', path='../models/yolov11n.pt')
        model.to(self.device)
        model.eval()
        self.get_logger().info("YOLOv11 model loaded.")
        return model

    def preprocess_image(self, cv_image):
        # Preprocess the image for YOLOv11
        image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)  # Convert to RGB
        image_tensor = transforms.ToTensor()(image_rgb).unsqueeze(0).to(self.device)
        return image_tensor

    def image_callback(self, msg):
        try:
            # Convert ROS 2 Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")
            return

        # Preprocess the image for YOLOv11
        input_tensor = self.preprocess_image(cv_image)

        # Perform object detection using YOLOv11
        with torch.no_grad():
            results = self.model(input_tensor)

        # Process and display detections
        self.process_detections(results, cv_image)

        # Display the image with detections
        cv2.imshow("YOLOv11 Object Detection", cv_image)
        cv2.waitKey(1)

    def process_detections(self, results, cv_image):
        # Extract detections
        detections = results.xyxy[0].cpu().numpy()

        # Draw bounding boxes and labels
        for detection in detections:
            x1, y1, x2, y2, conf, cls = detection
            label = self.classes[int(cls)]
            confidence = float(conf)

            if confidence > 0.5:  # Only show detections with confidence > 50%
                # Draw bounding box
                cv2.rectangle(cv_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                # Put label and confidence
                cv2.putText(cv_image, f"{label} {confidence:.2f}", (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        self.get_logger().info(f"Detected {len(detections)} objects.")

def main(args=None):
    rclpy.init(args=args)
    image_subscriber = ImageSubscriber()
    rclpy.spin(image_subscriber)
    image_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()