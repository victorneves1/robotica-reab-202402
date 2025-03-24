import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point, Pose, PoseStamped
from nav_msgs.msg import Odometry
import tf2_ros
import tf2_geometry_msgs
from visualization_msgs.msg import Marker, MarkerArray
from landmark_detector_interfaces.msg import LandmarkArray
from builtin_interfaces.msg import Duration


class LandmarkPublisher(Node):
    def __init__(self):
        super().__init__('landmark_publisher')
        
        # Create a publisher for landmarks
        self.landmark_publisher = self.create_publisher(MarkerArray, '/marker_array', 10)
        
        # Create a subscription to the odometry topic to get robot pose
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        
        # Create a subscription to landmarks
        self.create_subscription(LandmarkArray, '/landmarks', self.landmarks_callback, 10)

        # Create a tf2 listener to transform coordinates
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        self.robot_pose = None

    def odom_callback(self, msg):
        # Store the current robot pose from the odometry
        self.robot_pose = msg.pose.pose

    def landmarks_callback(self, msg: LandmarkArray):
        if self.robot_pose is None:
            self.get_logger().warn("No robot pose available yet.")
            return
        
        # Process the landmarks
        landmark_array = MarkerArray()
        for i, landmark in enumerate(msg.landmarks):
            marker = Marker()
            marker.header.frame_id = 'map'  # or use 'odom' if you'd like the marker in that frame
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = 'landmark'
            marker.id = i
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            marker.lifetime = Duration(sec=0, nanosec=0)
            marker.scale.x = 0.1
            marker.scale.y = 0.1
            marker.scale.z = 0.1
            marker.color.r = 1.0  # Red
            marker.color.g = 0.0  # Green
            marker.color.b = 0.0  # Blue
            marker.color.a = 1.0  # Alpha (transparency)

            # Here we assume the landmark message has x, y, z coordinates
            x, y, z = landmark.x_min, landmark.y_min, landmark.z_min  # Use appropriate coordinates
            
            # Create a PoseStamped for transforming landmark position to the robot frame
            landmark_pose = PoseStamped()
            landmark_pose.header.frame_id = "map"  # This could also be 'odom' depending on your reference frame
            landmark_pose.header.stamp = self.get_clock().now().to_msg()
            landmark_pose.pose.position.x = x
            landmark_pose.pose.position.y = y
            landmark_pose.pose.position.z = 0.0  # No z coordinate for this marker
            landmark_pose.pose.orientation.w = 1.0  # No rotation for this marker

            try:
                # Transform the landmark pose to the robot's frame (e.g., 'odom')
                transform = self.tf_buffer.lookup_transform('odom', landmark_pose.header.frame_id, rclpy.time.Time())
                transformed_pose = tf2_geometry_msgs.do_transform_pose(landmark_pose.pose, transform)

                # Set the marker position to the transformed coordinates
                marker.pose.position = transformed_pose.position
                marker.pose.orientation = transformed_pose.orientation

                # Add the marker to the array
                landmark_array.markers.append(marker)

            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
                self.get_logger().warn(f"Transform failed: {e}")
        
        # Publish the marker array to RViz
        self.landmark_publisher.publish(landmark_array)
        self.get_logger().info(f"Published {len(landmark_array.markers)} markers.")
        if len(landmark_array.markers) > 0:
            self.get_logger().info(f"First marker: {landmark_array.markers[0]}")
        

def main():
    rclpy.init()
    node = LandmarkPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
