import mediapipe as mp
import numpy as np
import cv2
import time


class HandDetector:
    def __init__(self, min_num_hands=1):
        self.min_num_hands = min_num_hands
        self.result = mp.tasks.vision.HandLandmarkerResult
        self.landmarker = mp.tasks.vision.HandLandmarker

        self.MARGIN = 10  # pixels
        self.FONT_SIZE = 1
        self.FONT_THICKNESS = 1
        self.HANDEDNESS_TEXT_COLOR = (88, 205, 54)  # vibrant green

        self.create_landmarker()

    def create_landmarker(self):
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path="./hand_landmarker.task"
            ),  # path to model
            running_mode=mp.tasks.vision.RunningMode.VIDEO,  # running on a live stream
            num_hands=self.min_num_hands,  # track both hands
            min_hand_detection_confidence=0.3,  # lower than value to get predictions more often
            min_hand_presence_confidence=0.3,  # lower than value to get predictions more often
            min_tracking_confidence=0.3,  # lower than value to get predictions more often
        )

        self.landmarker = self.landmarker.create_from_options(options)

    def detect_hands(self, image):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        self.result = self.landmarker.detect_for_video(
            mp_image, int(time.time() * 1000)
        )

    def draw_landmarks_on_image(self, rgb_image, detection_result):
        hand_landmarks_list = detection_result.hand_landmarks
        handedness_list = detection_result.handedness
        annotated_image = np.copy(rgb_image)

        # Loop through the detected hands to visualize.
        for idx in range(len(hand_landmarks_list)):
            hand_landmarks = hand_landmarks_list[idx]
            handedness = handedness_list[idx]

            # Draw the hand landmarks.
            mp.tasks.vision.drawing_utils.draw_landmarks(
                annotated_image,
                hand_landmarks,
                mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS,
                mp.tasks.vision.drawing_styles.get_default_hand_landmarks_style(),
                mp.tasks.vision.drawing_styles.get_default_hand_connections_style(),
            )

            # Get the top left corner of the detected hand's bounding box.
            height, width, _ = annotated_image.shape
            x_coordinates = [landmark.x for landmark in hand_landmarks]
            y_coordinates = [landmark.y for landmark in hand_landmarks]
            text_x = int(min(x_coordinates) * width)
            text_y = int(min(y_coordinates) * height) - self.MARGIN

            # Draw handedness (left or right hand) on the image.
            cv2.putText(
                annotated_image,
                f"{handedness[0].category_name}",
                (text_x, text_y),
                cv2.FONT_HERSHEY_DUPLEX,
                self.FONT_SIZE,
                self.HANDEDNESS_TEXT_COLOR,
                self.FONT_THICKNESS,
                cv2.LINE_AA,
            )

        return annotated_image

    def close(self):
        self.landmarker.close()
