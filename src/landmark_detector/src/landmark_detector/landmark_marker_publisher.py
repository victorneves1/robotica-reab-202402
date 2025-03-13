import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

# Assume your landmark message is defined in landmark_detector_interfaces/msg/Landmark
from landmark_detector_interfaces.msg import Landmark, LandmarkArray

class LandmarkMarkerPublisher(Node):
    def __init__(self):
        super().__init__('landmark_marker_publisher')
        # Publisher for visualization markers (RViz subscribes to /marker_array)
        self.marker_pub = self.create_publisher(MarkerArray, 'marker_array', 10)
        
        # Subscribe to the landmarks topic (where your detection node publishes)
        self.create_subscription(LandmarkArray, '/landmarks', self.landmarks_callback, 10)
        
        # We'll store the most recent landmarks
        self.latest_landmarks = []

    def landmarks_callback(self, msg: LandmarkArray):
        self.latest_landmarks = msg.landmarks
        self.publish_markers()

    def publish_markers(self):
        marker_array = MarkerArray()
        for i, lm in enumerate(self.latest_landmarks):
            marker = Marker()
            marker.header.frame_id = "odom"  # or the correct frame (e.g., "camera_link" or "odom")
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "landmark"
            marker.id = i
            marker.type = Marker.CUBE  # We'll draw a cube
            marker.action = Marker.ADD

            # Compute the center position from the axis-aligned bounding box
            marker.pose.position.x = (lm.x_min + lm.x_max) / 2.0
            marker.pose.position.y = (lm.y_min + lm.y_max) / 2.0
            marker.pose.position.z = (lm.z_min + lm.z_max) / 2.0

            # For axis-aligned boxes, orientation is identity (no rotation)
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0

            # The scale is the dimensions of the box
            marker.scale.x = lm.x_max - lm.x_min
            marker.scale.y = lm.y_max - lm.y_min
            marker.scale.z = lm.z_max - lm.z_min

            # Set the color (e.g., red with some transparency)
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 0.8

            # Optionally, set a lifetime for the marker so it disappears after a while
            marker.lifetime.sec = 1

            marker_array.markers.append(marker)
        
        self.marker_pub.publish(marker_array)
        self.get_logger().info(f"Published {len(marker_array.markers)} markers.")

def main():
    rclpy.init()
    node = LandmarkMarkerPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
