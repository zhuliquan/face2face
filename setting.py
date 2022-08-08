#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author : Administrator
# date   : 2018/6/21
############################ setting_dict for run webcam ############################
# detector                type: str/description: "dlib" or "face++"
# input_video_file        type: str/description: 0 or video file
# frozen_model_file       type: str/description: face2face model file
# enable_rotate_image     type: bool/description: whether rotate image for previous process
# enable_display_landmark type: bool/description: whether display landmark in final output video
# enable_output_video     type: bool/description: whether output final video
# output_video_file       type: str/description: output video file name
# output_video_coder      type: str/description: output video file coder method
# output_video_fps        type: int/description: output video file fps
# output_video_time       type: int/description: output video file time
# use_data_folder         type: str/description: replace face use previous data folder
# use_photo_folder        type: str/description: replace face use previous photo folder
# use_photo_number        type: int/description: replace face search range
# use_photo_gap           type: int/description: replace face search step
# enable_multiprocess     type: bool/description: whether use multiprocess to replace image
# use_core_number      type: int/description: replace process use multi-process core number
############################ setting_dict for generate data ###########################
# output_data_folder      type: str/description: generate previous data folder
# output_photo_folder     type: str/description: generate previous photo folder
# output_photo_number     type: int/description: generate previous photo number
# output_photo_gap        type: int/description: generate previous photo step
# output_frequency        type: int/description: save previous data frequency
# original_folder         type: str/description: generate train data for original images
# landmark_folder         type: str/description: generate train data for landmark images
#######################################################################################

setting_dict = {
    "detector_mode": "dlib",
    "input_video_file": "./input_video/input_final2.mp4",
    "frozen_model_file": "./frozen_model/frozen_model_replace_8000_150.pb",
    "enable_rotate_image": False,
    "enable_display_landmark": False,
    "enable_output_video": True,
    "output_video_file": "./output_video/tr2zl",  # tip:don't touch file extended type
    "output_video_coder": "MJPG",
    "output_video_fps": 30,
    "output_video_time": 10,
    "use_data_folder": "./preprocess_data/landmark_data",
    "use_photo_folder": "./preprocess_data/photo_data",
    "use_photo_number": 20000,
    "use_photo_gap": 1,
    "enable_multiprocess": False,
    "use_core_number": 4,
    "output_data_folder": "./preprocess_data/landmark_data",
    "output_photo_folder": "./preprocess_data/photo_data",
    "output_photo_number": 20000,
    "output_photo_gap": 1,
    "output_frequency": 1000,
    "original_folder": "./train_model/original",
    "landmark_folder": "./train_model/landmark",
}


class Setting:
    def __init__(self, args):
        self.detector_mode = args["detector_mode"]
        self.input_video_file = args["input_video_file"]
        self.frozen_model_file = args["frozen_model_file"]
        self.enable_rotate_image = args["enable_rotate_image"]
        self.enable_display_landmark = args["enable_display_landmark"]
        self.enable_output_video = args["enable_output_video"]
        self.output_video_file = args["output_video_file"]
        self.output_video_coder = args["output_video_coder"]
        self.output_video_fps = args["output_video_fps"]
        self.output_video_time = args["output_video_time"]
        self.use_data_folder = args["use_data_folder"]
        self.use_photo_folder = args["use_photo_folder"]
        self.use_photo_number = args["use_photo_number"]
        self.use_photo_gap = args["use_photo_gap"]
        self.enable_multiprocess = args["enable_multiprocess"]
        self.use_core_number = args["use_core_number"]
        self.output_data_folder = args["output_data_folder"]
        self.output_photo_folder = args["output_photo_folder"]
        self.output_photo_number = args["output_photo_number"]
        self.output_photo_gap = args["output_photo_gap"]
        self.output_frequency = args["output_frequency"]
        self.original_folder = args["original_folder"]
        self.landmark_folder = args["landmark_folder"]
