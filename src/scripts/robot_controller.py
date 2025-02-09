#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

class RobotController(Node):
    def __init__(self):
        super().__init__('robot_controller')
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        self.get_logger().info("Robot controller node started. Use WASD to move the robot, Q to stop, and E to exit.")

    def move_forward(self):
        twist = Twist()
        twist.linear.x = 1.0
        self.publisher_.publish(twist)
        self.get_logger().info("Moving forward")

    def move_backward(self):
        twist = Twist()
        twist.linear.x = -1.0
        self.publisher_.publish(twist)
        self.get_logger().info("Moving backward")

    def move_left(self):
        twist = Twist()
        twist.angular.z = 1.0
        self.publisher_.publish(twist)
        self.get_logger().info("Turning left")

    def move_right(self):
        twist = Twist()
        twist.angular.z = -1.0
        self.publisher_.publish(twist)
        self.get_logger().info("Turning right")

    def stop(self):
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.publisher_.publish(twist)
        self.get_logger().info("Stopping")

def main(args=None):
    rclpy.init(args=args)
    controller = RobotController()

    try:
        while True:
            command = input("Enter command (WASD to move, Q to stop, E to exit): ").strip().lower()
            if command == 'w':
                controller.move_forward()
            elif command == 's':
                controller.move_backward()
            elif command == 'a':
                controller.move_left()
            elif command == 'd':
                controller.move_right()
            elif command == 'q':
                controller.stop()
            elif command == 'e':
                break
            else:
                print("Invalid command. Use WASD to move, Q to stop, and E to exit.")
    except KeyboardInterrupt:
        pass
    finally:
        controller.stop()
        controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()