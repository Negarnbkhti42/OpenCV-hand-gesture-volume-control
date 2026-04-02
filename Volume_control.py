import cv2
import numpy as np
import time
import mediapipe as mp

cap = cv2.VideoCapture(0)
pastTime = 0

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img = cv2.resize(img, (640, 480))

    currentTime = time.time()
    fps = 1 / (currentTime - pastTime)
    pastTime = currentTime
    cv2.putText(
        img, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_PLAIN, 1, (21, 171, 91), 2
    )

    cv2.imshow("Img", img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
