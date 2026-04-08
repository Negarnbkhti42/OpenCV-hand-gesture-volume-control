import cv2
import numpy as np
import math
import time
import mediapipe as mp
from HandDetector import HandDetector


def detect_pointer_and_thumb(hand_landmarks):
    if len(hand_landmarks) == 0:
        return None, None

    # Get the coordinates of the index finger tip and thumb tip
    index_finger_tip = hand_landmarks[0][8]
    thumb_tip = hand_landmarks[0][4]

    return index_finger_tip, thumb_tip


def calculate_distance(point1, point2):
    distance = math.hypot(point2.x - point1.x, point2.y - point1.y)
    return distance


def highlight_pointer_and_thumb(img, hand_landmarks):
    index, thumb = detect_pointer_and_thumb(hand_landmarks)
    if index is not None and thumb is not None:
        height, width, _ = img.shape
        index_x, index_y = int(index.x * width), int(index.y * height)
        thumb_x, thumb_y = int(thumb.x * width), int(thumb.y * height)

        # Draw circles on the index finger tip and thumb tip
        cv2.circle(img, (index_x, index_y), 15, (195, 18, 142), cv2.FILLED)
        cv2.circle(img, (thumb_x, thumb_y), 15, (195, 18, 142), cv2.FILLED)
        cv2.line(img, (index_x, index_y), (thumb_x, thumb_y), (195, 18, 142), 3)


def calculate_volume_level(distance, min_distance=0.02, max_distance=0.2):
    # Normalize the distance to a volume level between 0 and 1
    if distance < min_distance:
        return 0.0
    elif distance > max_distance:
        return 1.0
    else:
        return (distance - min_distance) / (max_distance - min_distance)


def main():

    cap = cv2.VideoCapture(0)
    past_time = 0

    detector = HandDetector()

    while True:
        _, img = cap.read()
        img = cv2.flip(img, 1)
        img = cv2.resize(img, (640, 480))

        detector.detect_hands(img)
        img = detector.draw_landmarks_on_image(img, detector.result)
        highlight_pointer_and_thumb(img, detector.result.hand_landmarks)

        current_time = time.time()
        fps = 1 / (current_time - past_time)
        past_time = current_time
        cv2.putText(
            img,
            f"FPS: {int(fps)}",
            (10, 30),
            cv2.FONT_HERSHEY_PLAIN,
            1,
            (21, 171, 91),
            2,
        )

        cv2.imshow("Img", img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
