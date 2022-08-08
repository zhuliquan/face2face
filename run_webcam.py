#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author : zlq16
# date   : 2018/6/11
import cv2
import pickle
import numpy as np
import tensorflow as tf
from utils import rotate
from utils import resize
from utils import get_replace_image
from utils import time_show_decorate
from utils import load_graph
from utils import gpu_select_config
from utils import get_rectangle_wh
from utils import get_relative_landmark
from utils import get_landmark_vector
from utils import get_similar_image_index
from utils import generate_detector
from constant import CROP_SIZE
from constant import RESIZE_RATIO
from setting import Setting
from setting import setting_dict


@time_show_decorate
def run(args):
    with open(args.use_data_folder+"/count", "r") as f:
        args.use_photo_number = int(f.readline())
    with open(args.use_data_folder + "/previous_original_landmarks", "rb") as f:
        previous_original_landmarks = pickle.load(f)
    with open(args.use_data_folder + "/previous_jaw_vectors", "rb") as f:
        previous_jaw_vectors = pickle.load(f)

    print("use {0} to generate video".format(args.input_video_file))
    print("use {0} photos to replace".format(args.use_photo_number))

    # create detector
    detector = generate_detector(args)

    # load graph model
    config = gpu_select_config()
    graph = load_graph(args.frozen_model_file)
    image_tensor = graph.get_tensor_by_name('image_tensor:0')
    output_tensor = graph.get_tensor_by_name('generate_output/output:0')
    sess = tf.Session(graph=graph, config=config)

    # video reader/writer
    cap = cv2.VideoCapture(args.input_video_file)
    video_coder = cv2.VideoWriter_fourcc(*args.output_video_coder)
    video_writer = cv2.VideoWriter(args.output_video_file+".avi",
                                   video_coder, args.output_video_fps,
                                   (CROP_SIZE * 2, CROP_SIZE))

    count = 0
    lower_bound, upper_bound = 0, args.use_photo_number
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
            print("detect face")
        else:
            print("don't detect face")
            continue

        # draw landmark's shape on black image
        original_landmark = detector.resize_landmark(original_landmark, RESIZE_RATIO)
        black_image = np.zeros(frame_image.shape, np.uint8)
        black_image = detector.draw_landmark(black_image, original_landmark)

        # cut face from image
        face_rectangle = detector.get_face_rectangle(original_landmark)
        cut_black_image = detector.cut_image(black_image, face_rectangle)
        cut_frame_image = detector.cut_image(frame_image, face_rectangle)

        # generate target face image
        cut_black_image = cv2.resize(cut_black_image, (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_AREA)
        cut_frame_image = cv2.resize(cut_frame_image, (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_AREA)
        combined_image = np.concatenate([cut_black_image, cut_frame_image], axis=1)
        rgb_image = cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB)  # OpenCV uses BGR instead of RGB
        generated_image = sess.run(output_tensor, feed_dict={image_tensor: rgb_image})
        bgr_image = cv2.cvtColor(np.squeeze(generated_image), cv2.COLOR_RGB2BGR)

        # get more similar image
        relative_landmark = get_relative_landmark(original_landmark, face_rectangle)
        jaw_landmark = detector.get_jaw_landmark(relative_landmark)
        jaw_vector = get_landmark_vector(jaw_landmark)
        use_photo_range = (lower_bound, upper_bound, args.use_photo_gap)
        max_cosine, max_index = get_similar_image_index(use_photo_range, jaw_vector, previous_jaw_vectors)
        print(max_cosine, max_index)
        lower_bound, upper_bound = np.array([max_index - 20, max_index + 20]).clip(0, args.use_photo_number).tolist()
        replace_image = cv2.imread("{0}/{1}.jpg".format(args.use_photo_folder, max_index))
        replace_landmark = previous_original_landmarks[max_index]

        # use generate face image to replace similar image face
        w, h = get_rectangle_wh(face_rectangle)
        original_landmark[:, 0] = original_landmark[:, 0] - face_rectangle[0]
        original_landmark[:, 1] = original_landmark[:, 1] - face_rectangle[2]
        original_landmark[:, 0] = original_landmark[:, 0] * CROP_SIZE / w
        original_landmark[:, 1] = original_landmark[:, 1] * CROP_SIZE / h
        original_landmark = [tuple(p) for p in original_landmark]
        replace_landmark = [tuple(p) for p in replace_landmark]
        replace_image = get_replace_image(bgr_image, replace_image, original_landmark, replace_landmark)

        # concatenate original image and replace image for display
        black_image = resize(black_image)
        frame_image = resize(frame_image)
        replace_image = resize(replace_image)
        normal_image = np.concatenate([frame_image, replace_image], axis=1)
        landmark_image = np.concatenate([black_image, replace_image], axis=1)

        if args.enable_display_landmark:
            display_image = landmark_image
        else:
            display_image = normal_image

        cv2.imshow("frame", display_image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

        if args.enable_output_video:
            count += 1
            video_writer.write(display_image)

        if args.enable_output_video and count == args.output_video_fps * args.output_video_time:
            break

    sess.close()
    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    args = Setting(setting_dict)
    run(args)

