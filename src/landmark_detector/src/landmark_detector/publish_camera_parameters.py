import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import Header

class CameraInfoPublisher(Node):
    def __init__(self):
        super().__init__('camera_info_publisher')

        self.rgb_pub = self.create_publisher(CameraInfo, '/color/camera_info', 10)
        self.depth_pub = self.create_publisher(CameraInfo, '/depth/camera_info', 10)

        self.timer = self.create_timer(0.03, self.publish)  # 30 Hz

    def publish(self):
        timestamp = self.get_clock().now().to_msg()
        header = Header()
        header.stamp = timestamp
        header.frame_id = 'camera_link'

        # Shared intrinsics for simplicity
        width = 960
        height = 540
        fx = float(615.0)
        fy = float(615.0)
        cx = float(480.0)
        cy = float(270.0)
        
        cam_info = CameraInfo()
        cam_info.header = header
        cam_info.height = height
        cam_info.width = width
        cam_info.distortion_model = 'plumb_bob'
        cam_info.d = [0.0, 0.0, 0.0, 0.0, 0.0]  # No distortion

        cam_info.r = [1.0, 0.0, 0.0,
                      0.0, 1.0, 0.0,
                      0.0, 0.0, 1.0]
        
        cam_info.k = [fx, 0.0, cx,
                      0.0, fy, cy,
                      0.0, 0.0, 1.0]

        cam_info.p = [fx, 0.0, cx, 0.0,
                      0.0, fy, cy, 0.0,
                      0.0, 0.0, 1.0, 0.0]


        self.rgb_pub.publish(cam_info)
        self.depth_pub.publish(cam_info)

def main():
    rclpy.init()
    node = CameraInfoPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()