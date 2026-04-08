import cv2
import numpy as np
import math
import time
from pycaw.pycaw import AudioUtilities
import mediapipe as mp
from HandDetector import HandDetector


def get_point_position(img, point):
    height, width, _ = img.shape
    x = int(point.x * width)
    y = int(point.y * height)
    return {"x": x, "y": y}


def detect_pointer_and_thumb(hand_landmarks):
    if len(hand_landmarks) == 0:
        return None, None

    # Get the coordinates of the index finger tip and thumb tip
    index_finger_tip = hand_landmarks[0][8]
    thumb_tip = hand_landmarks[0][4]

    return index_finger_tip, thumb_tip


def calculate_distance(point1, point2):
    distance = math.hypot(point2["x"] - point1["x"], point2["y"] - point1["y"])
    return distance


def highlight_pointer_and_thumb(img, index, thumb):
    # Draw circles on the index finger tip and thumb tip
    cv2.circle(img, (index["x"], index["y"]), 15, (195, 18, 142), cv2.FILLED)
    cv2.circle(img, (thumb["x"], thumb["y"]), 15, (195, 18, 142), cv2.FILLED)
    cv2.line(img, (index["x"], index["y"]), (thumb["x"], thumb["y"]), (195, 18, 142), 3)


def set_system_volume(volume_level):
    device = AudioUtilities.GetSpeakers()
    volume = device.EndpointVolume
    volume.SetMasterVolumeLevel(volume_level, None)


def main():

    cap = cv2.VideoCapture(0)
    past_time = 0

    detector = HandDetector()

    device = AudioUtilities.GetSpeakers()
    volume = device.EndpointVolume
    print(f"Audio output: {device.FriendlyName}")
    print(f"- Muted: {bool(volume.GetMute())}")
    print(f"- Volume level: {volume.GetMasterVolumeLevel()} dB")
    print(
        f"- Volume range: {volume.GetVolumeRange()[0]} dB - {volume.GetVolumeRange()[1]} dB"
    )

    while True:
        _, img = cap.read()
        img = cv2.flip(img, 1)
        img = cv2.resize(img, (640, 480))

        detector.detect_hands(img)
        img = detector.draw_landmarks_on_image(img, detector.result)
        index_finger_tip, thumb_tip = detect_pointer_and_thumb(
            detector.result.hand_landmarks
        )

        if index_finger_tip is not None and thumb_tip is not None:
            # get the actual pixel positions of the index finger tip and thumb tip
            index_finger_tip = get_point_position(img, index_finger_tip)
            thumb_tip = get_point_position(img, thumb_tip)

            # Highlight the pointer and thumb
            highlight_pointer_and_thumb(img, index_finger_tip, thumb_tip)

            # Set volume based on the distance between the index finger tip and thumb tip
            distance = calculate_distance(index_finger_tip, thumb_tip)
            volume_level = np.interp(
                distance,
                [50, 250],
                [volume.GetVolumeRange()[0], volume.GetVolumeRange()[1]],
            )  # Map distance to volume level
            print(f"Distance: {distance:.2f}, Volume level: {volume_level:.2f} dB")
            set_system_volume(volume_level)

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
