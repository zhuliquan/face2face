#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author : zlq16
# date   : 2018/6/10
import os
import cv2
import pickle
from imutils import video
from utils import rotate
from setting import Setting
from setting import setting_dict
from utils import get_landmark_vector
from utils import get_relative_landmark
from utils import generate_detector
from constant import RESIZE_RATIO


def save_data():
    with open(args.output_data_folder + "/count", "w") as f:
        f.write(str(count))
    with open("{0}/{1}".format(args.output_data_folder, "previous_original_landmarks"), "wb") as f:
        pickle.dump(previous_original_landmarks, f)
    with open("{0}/{1}".format(args.output_data_folder, "previous_face_rectangles"), "wb") as f:
        pickle.dump(previous_face_rectangles, f)
    with open("{0}/{1}".format(args.output_data_folder, "previous_jaw_vectors"), "wb") as f:
        pickle.dump(previous_jaw_vectors, f)
    with open("{0}/{1}".format(args.output_data_folder, "previous_landmark_vectors"), "wb") as f:
        pickle.dump(previous_landmark_vectors, f)


if __name__ == '__main__':
    args = Setting(setting_dict)
    print("previous process {0}".format(args.input_video_file))
    print("photo save {0}".format(args.output_photo_folder))
    print("data save {0}".format(args.output_data_folder))

    # create detector
    detector = generate_detector(args)

    # make dir
    os.makedirs(args.output_photo_folder, exist_ok=True)
    os.makedirs(args.output_data_folder, exist_ok=True)

    # video reader
    cap = cv2.VideoCapture(args.input_video_file)
    fps = video.FPS().start()

    timer, count = 0, 0
    previous_original_landmarks = []
    previous_face_rectangles = []
    previous_jaw_vectors = []
    previous_landmark_vectors = []

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
                cv2.imwrite("{0}/{1}.jpg".format(args.output_photo_folder, count-1), frame_resize)
                print("{0} detect face".format(count))
            else:
                continue
        else:
            print("don't detect face")
            continue

        # previous process data
        face_rectangle = detector.get_face_rectangle(original_landmark)
        relative_landmark = get_relative_landmark(original_landmark, face_rectangle)
        jaw_vector = get_landmark_vector(detector.get_jaw_landmark(relative_landmark))
        landmark_vector = get_landmark_vector(relative_landmark)

        # save image data
        previous_original_landmarks.append(original_landmark)
        previous_face_rectangles.append(face_rectangle)
        previous_jaw_vectors.append(jaw_vector)
        previous_landmark_vectors.append(landmark_vector)

        if count % args.output_frequency == 0:
            save_data()

        if count >= args.output_photo_number:
            break
        fps.update()

    fps.stop()
    print("total generate {0} photo previous data".format(count))
    print('[INFO] elapsed time (total): {:.2f}'.format(fps.elapsed()))
    print('[INFO] approx. FPS: {:.2f}'.format(fps.fps()))
    cap.release()
    cv2.destroyAllWindows()
    save_data()
