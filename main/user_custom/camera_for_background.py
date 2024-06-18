import cv2
from time import sleep

#get the first bg frame
cap = cv2.VideoCapture(0)
sleep(5)
ret, frame = cap.read()
cv2.imwrite('/home/pi/Workspace/bg.jpg', frame)
