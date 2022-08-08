import cv2

cap = cv2.VideoCapture("./input_final.mp4")
count = 1
while True:
    ret, frame = cap.read()
    cv2.imwrite("{0}.jpg".format(count), frame)
    if count == 5:
        break
    count += 1