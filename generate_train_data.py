#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author : zlq16
# date   : 2018/6/10
import os
import cv2
import numpy as np
from imutils import video
from utils import rotate
from setting import setting_dict
from setting import Setting
from constant import RESIZE_RATIO
from utils import generate_detector
if __name__ == '__main__':
    args = Setting(setting_dict)
    print("generate face train image from {0}".format(args.input_video_file))
    print("save original image on {0}".format(args.original_folder))
    print("save landmark image on {0}".format(args.landmark_folder))

    # create detector
    detector = generate_detector(args)

    # make dir
    os.makedirs(args.original_folder, exist_ok=True)
    os.makedirs(args.landmark_folder, exist_ok=True)

    # video reader
    cap = cv2.VideoCapture(args.input_video_file)
    fps = video.FPS().start()

    timer, count = 0, 0
    while cap.isOpened():
        ret, frame_image = cap.read()
        if not ret:
            break
        if args.enable_rotate_image:
            frame_image = rotate(frame_image, 270)

        # get landmark's points
        frame_resize = cv2.resize(frame_image, None, fx=1 / RESIZE_RATIO, fy=1 / RESIZE_RATIO)
        gray_image = cv2.cvtColor(frame_resize, cv2.COLOR_BGR2GRAY)
        original_landmark, flag = detector.get_landmark(gray_image)

        if flag:
            timer += 1
            if timer == args.output_photo_gap:
                timer = 0
                count += 1
                print("{0} detect face".format(count))
            else:
                continue
        else:
            print("don't detect face")
            continue

        # draw landmark's shape on black image
        black_image = np.zeros(frame_image.shape, np.uint8)
        face_rectangle = detector.get_face_rectangle(original_landmark)
        original_landmark = detector.resize_landmark(original_landmark, RESIZE_RATIO)
        face_rectangle = detector.resize_face_rectangle(face_rectangle, RESIZE_RATIO)
        black_image = detector.draw_landmark(black_image, original_landmark)

        # cut face from image
        cut_black_image = detector.cut_image(black_image, face_rectangle)
        cut_frame_image = detector.cut_image(frame_image, face_rectangle)

        # save image
        cv2.imwrite("{0}/{1}.png".format(args.landmark_folder, count), cut_black_image)
        cv2.imwrite("{0}/{1}.png".format(args.original_folder, count), cut_frame_image)

        if count == args.output_photo_number:
            break
        elif cv2.waitKey(10) & 0xFF == ord('q'):
            break

        fps.update()

    fps.stop()
    print('[INFO] elapsed time (total): {:.2f}'.format(fps.elapsed()))
    print('[INFO] approx. FPS: {:.2f}'.format(fps.fps()))
    cap.release()
    cv2.destroyAllWindows()
