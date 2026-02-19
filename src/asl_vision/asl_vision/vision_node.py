import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math

class ASLVisionNode(Node):
    def __init__(self):
        super().__init__('asl_vision_node')

        # subscribe to our camera node
        self.subscription = self.create_subscription(
                Image,
                'image_raw',
                self.listener_callback,
                10)

        self.bridge = CvBridge()

        # init mediapipe landmarker
        base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
        options = vision.HandLandmarkerOptions(base_options=base_options,
                                               num_hands=1,
                                               min_hand_detection_confidence=0.5,
                                               min_hand_presence_confidence=0.5,
                                               min_tracking_confidence=0.5)
        self.detector = vision.HandLandmarker.create_from_options(options)

        self.last_prediction = "None"
        self.last_landmarks = None

        self.frame_count = 0
    
    def listener_callback(self, msg):
        try:
            self.frame_count += 1
            one_frame_skipped = self.frame_count % 2 != 0

            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            frame = cv2.resize(frame, (320, 240)) # (optimization)

            if one_frame_skipped: # (optimization)

                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                detection_result = self.detector.detect(mp_image)

                if detection_result.hand_landmarks:
                    self.last_landmarks = detection_result.hand_landmarks[0]
                    self.last_prediction = self.classify_gesture(self.last_landmarks)
                else:
                    self.last_landmarks = None
                    self.last_prediction = "None"

            # draw green dots
            GREEN = (0, 255, 0)
            RADIUS = 3
            h, w, _ = frame.shape
            if self.last_landmarks:
                for lm in self.last_landmarks:
                    cx, cy, = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (cx, cy), RADIUS, GREEN, -1)

            # draw prediction
            cv2.putText(frame, f'Sign: {self.last_prediction}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, GREEN, 2)
            cv2.imshow("ASL Recognition", frame)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f"error: {str(e)}")

    def get_dist(self, p1, p2):
        return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

    def classify_gesture(self, landmarks):
        WRIST = landmarks[0]
        THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = landmarks[1], landmarks[2], landmarks[3], landmarks[4]
        INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = landmarks[5], landmarks[6], landmarks[7], landmarks[8]
        MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = landmarks[9], landmarks[10], landmarks[11], landmarks[12]
        RING_MCP, RING_PIP, RING_DIP, RING_TIP = landmarks[13], landmarks[14], landmarks[15], landmarks[16]
        PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = landmarks[17], landmarks[18], landmarks[19], landmarks[20]

        # is each finger pointing up?
        i = INDEX_TIP.y < INDEX_PIP.y
        m = MIDDLE_TIP.y < MIDDLE_PIP.y
        r = RING_TIP.y < RING_PIP.y
        p = PINKY_TIP.y < PINKY_PIP.y

        # fist signs
        if not any([i, m, r, p]):
            if THUMB_CMC.y > THUMB_MCP.y > THUMB_IP.y > THUMB_TIP.y:
                return "A"


        return "Unable to classify gesture"


def main(args=None):
    rclpy.init(args=args)
    node = ASLVisionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
