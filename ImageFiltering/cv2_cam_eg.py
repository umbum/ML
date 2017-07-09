import cv2
import numpy as np


cap = cv2.VideoCapture(0)
while True:
        ret, frame = cap.read()
        hsv= cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_blue = np.array([0, 50, 50])
        upper_blue = np.array([20, 255, 255])
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        res = cv2.bitwise_and(frame, frame, mask=mask)
        cv2.imshow('image', frame)
        cv2.imshow('mask', mask)
        cv2.imshow('res', res)
        # waitKey(0)을 넣으면 block이 걸리기 때문에 1ms를 준다.
        if cv2.waitKey(1) & 0xFF == ord('q'):
                break
cap.release()
cv2.destroyAllWindows()