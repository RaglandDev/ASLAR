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
        tips = [8, 12, 16, 20]
        pips = [6, 10, 14, 18]
        up = []
        for t, p in zip(tips, pips):
            up.append(landmarks[t].y < landmarks[p].y)
        i_up, m_up, r_up, p_up = up

        thumb_out = landmarks[4].x > landmarks[5].x

        # index and middle crossed
        fingers_crossed = landmarks[8].x < landmarks[12].x 

        # thumb is touching fingers
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        dist_thumb_index = self.get_dist(thumb_tip, index_tip)

        # (thumb, index, middle, ring, pinky)
        state = (thumb_out, *up)
        asl_map = {
                (False, True, False, False, False): "D",
                (True, True, False, False, False): "L",
                (False, True, True, False, False): "V",
                (False, True, True, True, False): "W",
                (False, False, False, False, True): "I",
                (True, False, False, False, True): "Y",
                (False, True, True, True, True): "B",
                (False, False, False, False, False): "S/E/M/N", # fist
                (False, False, True, True, True): "F",
                (True, True, True, False, False): "K",
                (False, True, True, False, False): "R" if fingers_crossed else "V",
                (True, False, False, False, False): "A",
            }

        DISTANCE_THRESHOLD = 0.05
        if all(landmarks[t].y > landmarks[p].y for t, p in zip(tips, pips)) and dist_thumb_index < DISTANCE_THRESHOLD:
            return "O"

        return asl_map.get(state, "None")


def main(args=None):
    rclpy.init(args=args)
    node = ASLVisionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
