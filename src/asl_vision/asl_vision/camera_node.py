import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge

class CameraNode(Node):
    def __init__(self):
        super().__init__('camera_node')

        # create pub
        self.publisher_ = self.create_publisher(Image, 'image_raw', 10)

        # set timer
        timer_period = 0.033 # 30fps -> a frame every 33ms
        self.timer = self.create_timer(timer_period, self.timer_callback)

        # init OpenCV camera and bridge
        self.cap = cv2.VideoCapture(2)
        self.bridge = CvBridge()

    def timer_callback(self):
        ret, frame = self.cap.read()
        if ret:
            # translate OpenCV 'frame' into a ROS2 message
            msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")

            msg.header.frame_id = 'camera_link'

            # publish message to 'image_raw' topic
            self.publisher_.publish(msg)
            self.get_logger().info('Publishing video frame')

def main(args=None):
    rclpy.init(args=args)

    node = CameraNode()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
