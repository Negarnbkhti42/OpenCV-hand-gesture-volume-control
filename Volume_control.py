import cv2
import numpy as np
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


def highlight_pointer_and_thumb(img, hand_landmarks):
    index, thumb = detect_pointer_and_thumb(hand_landmarks)
    if index is not None and thumb is not None:
        height, width, _ = img.shape
        index_x, index_y = int(index.x * width), int(index.y * height)
        thumb_x, thumb_y = int(thumb.x * width), int(thumb.y * height)

        # Draw circles on the index finger tip and thumb tip
        cv2.circle(img, (index_x, index_y), 15, (195, 18, 142), cv2.FILLED)
        cv2.circle(img, (thumb_x, thumb_y), 15, (195, 18, 142), cv2.FILLED)


def main():

    cap = cv2.VideoCapture(0)
    past_time = 0

    detector = HandDetector()

    while True:
        _, img = cap.read()
        img = cv2.flip(img, 1)
        img = cv2.resize(img, (640, 480))

        detector.detect_hands(img)
        highlight_pointer_and_thumb(img, detector.result.hand_landmarks)
        img = detector.draw_landmarks_on_image(img, detector.result)

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
