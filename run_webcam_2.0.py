#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author : Administrator
# date   : 2018/6/20
import pickle
from utils import time_show_decorate
from utils import generate_detector
from utils import previous_process_video
from utils import generate_face_image_use_network
from utils import replace_process
from utils import operator_final_video
from setting import setting_dict
from setting import Setting


@time_show_decorate
def run():
    print("use {0} to generate video".format(args.input_video_file))
    print("use {0} photos to replace".format(args.use_photo_number))

    with open(args.use_data_folder+"/count", "r") as f:
        args.use_photo_number = int(f.readline())
    with open(args.use_data_folder + "/previous_original_landmarks", "rb") as f:
        previous_original_landmarks = pickle.load(f)
    with open(args.use_data_folder + "/previous_landmark_vectors", "rb") as f:
        previous_landmark_vectors = pickle.load(f)

    # video previous process
    print("previous process images")
    detector = generate_detector(args)
    rgb_images, black_images, frame_images, face_rectangles, original_landmarks = previous_process_video(
        args, detector)

    # generate face image
    print("process generate images")
    face_images = generate_face_image_use_network(args, rgb_images)

    # replace face image
    print("process replace images")
    image_number = len(face_images)
    display_images = replace_process(
        args, 0, image_number,
        face_images, black_images, frame_images,
        face_rectangles, original_landmarks,
        previous_landmark_vectors, previous_original_landmarks
    )

    # save video or show video
    print("show video or save video")
    operator_final_video(args, display_images)


if __name__ == '__main__':
    args = Setting(setting_dict)
    run()
