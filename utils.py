#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author : Administrator
# date   : 2018/6/20
import os
import cv2
import time
import dlib
import numpy as np
from constant import GPU_ID
from constant import CROP_SIZE
from constant import RESIZE_RATIO
from constant import LANDMARKS_SHAPE_FILE


#### decorate ####
def time_show_decorate(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print("elapse {0} s".format(end_time - start_time))
        return result
    return wrapper


#### about face detector ####
def reshape_for_polyline(array):
    """reshape for ploy"""
    return np.array(array, np.int32).reshape((-1, 1, 2))


class Detector:
    def __init__(self):
        pass

    def get_landmark(self, image):
        pass

    def get_jaw_landmark(self, landmark):
        pass

    def draw_landmark(self, black_image, landmark):
        pass

    @staticmethod
    def draw_points(black_image, landmarks):
        if len(landmarks) != 0:
            shape = black_image.shape
            if len(shape) == 3:
                color = (255, 255, 255)
            else:
                color = (255, 255)
            thickness = 2
            for point in landmarks:
                cv2.circle(black_image, tuple(point), thickness, color, thickness)
        return black_image

    @staticmethod
    def get_face_rectangle(landmarks):
        if len(landmarks) != 0:
            x_min = np.min(landmarks[:, 0])
            x_max = np.max(landmarks[:, 0])
            y_min = np.min(landmarks[:, 1])
            y_max = np.max(landmarks[:, 1])
            face_rectangle = [x_min, x_max, y_min, y_max]
        else:
            face_rectangle = []
        face_rectangle = np.array(face_rectangle, np.int32)
        return face_rectangle

    @staticmethod
    def resize_landmark(landmarks, ratio):
        if len(landmarks) != 0:
            return landmarks * ratio
        return landmarks

    @staticmethod
    def resize_face_rectangle(face_rectangle, ratio):
        if len(face_rectangle) != 0:
            return face_rectangle * ratio
        return face_rectangle

    @staticmethod
    def cut_image(image, rectangle):
        if len(rectangle) == 0:
            cut_image = image
        else:
            shape = image.shape
            if len(shape) == 3:
                cut_image = image[rectangle[2]:rectangle[3] + 1, rectangle[0]:rectangle[1] + 1, :]
            else:
                cut_image = image[rectangle[2]:rectangle[3] + 1, rectangle[0]:rectangle[1] + 1]
        return cut_image


class DLibDetector(Detector):
    def __init__(self):
        super(DLibDetector, self).__init__()
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(LANDMARKS_SHAPE_FILE)

    def get_landmark(self, image):
        faces = self.detector(image, 1)
        flag = False
        if len(faces) != 0:
            face = faces[0]
            landmarks = self.predictor(image, face).parts()
            landmarks = [[p.x, p.y] for p in landmarks]
            flag = True
        else:
            landmarks = []
        landmarks = np.array(landmarks)
        return landmarks, flag

    def get_jaw_landmark(self, landmark):
        return landmark[0:17]

    def draw_landmark(self, black_image, landmark):
        if len(landmark) != 0:
            shape = black_image.shape
            if len(shape) == 3:
                color = (255, 255, 255)
            else:
                color = (255, 255)
            thickness = 2
            jaw = reshape_for_polyline(landmark[0:17])
            left_eyebrow = reshape_for_polyline(landmark[22:27])
            right_eyebrow = reshape_for_polyline(landmark[17:22])
            nose_bridge = reshape_for_polyline(landmark[27:31])
            lower_nose = reshape_for_polyline(landmark[30:35])
            left_eye = reshape_for_polyline(landmark[42:48])
            right_eye = reshape_for_polyline(landmark[36:42])
            outer_lip = reshape_for_polyline(landmark[48:60])
            inner_lip = reshape_for_polyline(landmark[60:68])
            cv2.polylines(black_image, [jaw], False, color, thickness)
            cv2.polylines(black_image, [left_eyebrow], False, color, thickness)
            cv2.polylines(black_image, [right_eyebrow], False, color, thickness)
            cv2.polylines(black_image, [nose_bridge], False, color, thickness)
            cv2.polylines(black_image, [lower_nose], True, color, thickness)
            cv2.polylines(black_image, [left_eye], True, color, thickness)
            cv2.polylines(black_image, [right_eye], True, color, thickness)
            cv2.polylines(black_image, [outer_lip], True, color, thickness)
            cv2.polylines(black_image, [inner_lip], True, color, thickness)
        return black_image


def generate_detector(args):
    if args.detector_mode == "dlib":
        detector = DLibDetector()
    else:
        raise Exception("检测模式选择有问题")
    return detector


#### about image process ####
def image_read(file_name):
    return cv2.imread(file_name)


def resize(image):
    """Crop and resize image for pix2pix."""
    height, width, _ = image.shape
    if height != width:
        # crop to correct ratio
        size = min(height, width)
        oh = (height - size) // 2
        ow = (width - size) // 2
        cropped_image = image[oh:(oh + size), ow:(ow + size), :]
    else:
        cropped_image = image
    image_resize = cv2.resize(cropped_image, (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_AREA)
    return image_resize


def rotate(image, angle, center=None, scale=1.0):
    """rotate image"""
    (h, w) = image.shape[:2]
    if center is None:
        center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated


def apply_affine_transform(src, src_tri, dst_tri, size):
    """Apply affine transform calculated using src_tri and dst_tri to src and output an image of size."""
    # Given a pair of triangles, find the affine transform.
    warp_mat = cv2.getAffineTransform(np.float32(src_tri), np.float32(dst_tri))

    # Apply the Affine Transform just found to the src image
    dst = cv2.warpAffine(
        src,
        warp_mat, (size[0], size[1]),
        None,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101)
    return dst


def rect_contains(rect, point):
    """Check if a point is inside a face_rectangle"""
    if point[0] < rect[0]:
        return False
    elif point[1] < rect[1]:
        return False
    elif point[0] > rect[0] + rect[2]:
        return False
    elif point[1] > rect[1] + rect[3]:
        return False
    return True


def calculate_delaunay_triangles(rect, points):
    """calculate delanauy triangle"""
    # create subdiv
    subdiv = cv2.Subdiv2D(rect)

    # Insert points into subdiv
    for p in points:
        subdiv.insert(p)

    triangle_list = subdiv.getTriangleList()

    delaunay_triangle = list()
    len_point = len(points)
    for t in triangle_list:
        pt = list()
        pt.append((t[0], t[1]))
        pt.append((t[2], t[3]))
        pt.append((t[4], t[5]))

        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])

        if rect_contains(rect, pt1) and rect_contains(rect, pt2) and rect_contains(rect, pt3):
            # Get face-points (from 68 face detector) by coordinates
            ind = [k for j in range(0, 3) for k in range(0, len_point) if (abs(pt[j][0] - points[k][0]) < 1.0 and abs(pt[j][1] - points[k][1]) < 1.0)]
            # Three points form a triangle. Triangle array corresponds to the file tri.txt in FaceMorph
            if len(ind) == 3:
                delaunay_triangle.append((ind[0], ind[1], ind[2]))
    return delaunay_triangle


def warp_triangle(image1, image2, t1, t2):
    """Warps and alpha blends triangular regions from image1 and image2 to image"""
    # Find bounding face_rectangle for each triangle
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))

    # Offset points by left top corner of the respective face_rectangle
    t1_rectangle = [((t1[i][0] - r1[0]), (t1[i][1] - r1[1])) for i in range(0, 3)]
    t2_rectangle = [((t2[i][0] - r2[0]), (t2[i][1] - r2[1])) for i in range(0, 3)]
    t2_rect_int = [((t2[i][0] - r2[0]), (t2[i][1] - r2[1])) for i in range(0, 3)]

    # Get mask by filling triangle
    mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2_rect_int), (1.0, 1.0, 1.0), 16, 0)

    # Apply warpImage to small rectangular patches
    image1_rect = image1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    size = (r2[2], r2[3])
    image2_rect = apply_affine_transform(image1_rect, t1_rectangle, t2_rectangle, size)
    image2_rect = image2_rect * mask

    # Copy triangular region of the rectangular patch to the output image
    image2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = image2[r2[
        1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] * ((1.0, 1.0, 1.0) - mask)

    image2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = image2[r2[
        1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] + image2_rect


def get_replace_image(image1, image2, landmarks1, landmarks2):
    """use image1 face to replace image2 face"""
    image1_warped = np.copy(image2)
    # Find convex hull
    hull_index = cv2.convexHull(np.array(landmarks1), returnPoints=False)
    hull1 = [landmarks1[int(p)] for p in hull_index]
    hull2 = [landmarks2[int(p)] for p in hull_index]

    # Find delanauy traingulation for convex hull points
    image2_size = image2.shape
    rect = (0, 0, image2_size[1], image2_size[0])
    dt = calculate_delaunay_triangles(rect, hull2)

    if len(dt) == 0:
        quit()

    # Apply affine transformation to Delaunay triangles
    len_dt = len(dt)
    for i in range(0, len_dt):
        # get points for image1, image2 corresponding to the triangles
        t1 = [hull1[dt[i][j]] for j in range(0, 3)]
        t2 = [hull2[dt[i][j]] for j in range(0, 3)]
        warp_triangle(image1, image1_warped, t1, t2)

    # Calculate Mask
    hull8U = [(p[0], p[1]) for p in hull2]

    mask = np.zeros(image2.shape, dtype=image2.dtype)

    cv2.fillConvexPoly(mask, np.int32(hull8U), (255, 255, 255))

    r = cv2.boundingRect(np.float32([hull2]))

    center = (r[0]+int(r[2]/2), r[1]+int(r[3]/2))

    # Clone seamlessly.
    output = cv2.seamlessClone(np.uint8(image1_warped), image2, mask, center, cv2.NORMAL_CLONE)
    return output


def get_relative_landmark(landmark, face_rectangle):
    x_ = (face_rectangle[0] + face_rectangle[1]) / 2
    y_ = (face_rectangle[2] + face_rectangle[3]) / 2
    relative_landmark = np.copy(landmark)
    relative_landmark[:, 0] = landmark[:, 0] - x_
    relative_landmark[:, 1] = landmark[:, 1] - y_
    return relative_landmark


def get_landmark_vector(landmark):
    return landmark.flatten()[:]


def get_rectangle_wh(rectangle):
    return (rectangle[1]-rectangle[0]), (rectangle[3]-rectangle[2])


def get_similar_image_index(use_photo_range, a_vector, previous_landmark_vectors):
    start_index, end_index, step = use_photo_range
    max_cosine = -1
    target_index = start_index
    for index in range(start_index, end_index, step):
        b_vector = previous_landmark_vectors[index]
        num = np.sum(a_vector * b_vector)
        norm = (np.linalg.norm(a_vector) * np.linalg.norm(b_vector))
        cos = num / norm
        if max_cosine < cos:
            max_cosine = cos
            target_index = index
    return max_cosine, target_index


def replace_process(args, process_start_index, process_end_index,
                    face_images, black_images, frame_images,
                    face_rectangles, original_landmarks,
                    previous_landmark_vectors, previous_original_landmarks):

    display_images = []
    for index in range(process_start_index, process_end_index):
        print(index)
        face_image = face_images[index]
        black_image = black_images[index]
        frame_image = frame_images[index]
        original_landmark = original_landmarks[index]
        face_rectangle = face_rectangles[index]

        # get more similar image
        relative_landmark = get_relative_landmark(original_landmark, face_rectangle)
        landmark_vector = get_landmark_vector(relative_landmark)
        use_photo_range = (0, args.use_photo_number, args.use_photo_gap)
        max_cosine, max_index = get_similar_image_index(use_photo_range,
                                                        landmark_vector,
                                                        previous_landmark_vectors)
        print(max_cosine, max_index)
        replace_image = image_read("{0}/{1}.jpg".format(args.use_photo_folder, max_index))
        replace_landmark = previous_original_landmarks[max_index]

        # use generate face image to replace similar image face
        w, h = get_rectangle_wh(face_rectangle)
        original_landmark[:, 0] = original_landmark[:, 0] - face_rectangle[0]
        original_landmark[:, 1] = original_landmark[:, 1] - face_rectangle[2]
        original_landmark[:, 0] = original_landmark[:, 0] * CROP_SIZE / w
        original_landmark[:, 1] = original_landmark[:, 1] * CROP_SIZE / h
        original_landmark = [tuple(p) for p in original_landmark]
        replace_landmark = [tuple(p) for p in replace_landmark]

        replace_image = get_replace_image(face_image, replace_image, original_landmark, replace_landmark)

        # concatenate original image and replace image for display
        black_image = resize(black_image)
        frame_image = resize(frame_image)
        replace_image = resize(replace_image)
        image_normal = np.concatenate([frame_image, replace_image], axis=1)
        image_landmark = np.concatenate([black_image, replace_image], axis=1)

        if args.display_landmark:
            display_images.append(image_landmark)
        else:
            display_images.append(image_normal)
    return display_images


#### about network ####
def gpu_select_config():
    import tensorflow as tf
    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    print("GPU {0} is selected".format(GPU_ID + 1))
    return config


def load_graph(frozen_graph_filename):
    """load frozen model file"""
    import tensorflow as tf
    graph = tf.Graph()
    with graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(frozen_graph_filename, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return graph


def generate_face_image_use_network(args, rgb_images):
    import tensorflow as tf
    face_images = []
    # import graph model
    config = gpu_select_config()
    graph = load_graph(args.frozen_model_file)
    image_tensor = graph.get_tensor_by_name('image_tensor:0')
    output_tensor = graph.get_tensor_by_name('generate_output/output:0')
    sess = tf.Session(graph=graph, config=config)
    fps = args.output_video_fps
    shape = image_tensor.get_shape()
    if len(shape) == 3:
        image_number = len(rgb_images)
        for i in range(image_number):
            print(i)
            generated_image = sess.run(output_tensor, feed_dict={image_tensor: rgb_images[i]})
            generated_image = cv2.cvtColor(np.squeeze(generated_image), cv2.COLOR_RGB2BGR)
            face_images.append(generated_image)
    else:
        for time in range(args.output_video_time):
            print("{0} s.".format(time))
            input_images = np.array(rgb_images[time * fps:(time + 1) * fps])
            generated_images = sess.run(output_tensor, feed_dict={image_tensor: input_images})
            generated_images = [cv2.cvtColor(np.squeeze(image), cv2.COLOR_RGB2BGR) for image in generated_images]
            face_images.extend(generated_images)
    sess.close()
    return face_images


#### about video process ####
def previous_process_video(args, detector):
    frame_images = []
    black_images = []
    face_rectangles = []
    original_landmarks = []
    rgb_images = []

    # camera object
    cap = cv2.VideoCapture(args.input_video_file)

    count = 0
    while cap.isOpened():
        # read frame
        ret, frame_image = cap.read()
        if not ret:
            break
        if args.enable_rotate_image:
            frame_image = rotate(frame_image, 270)

        # get landmark's points
        frame_resize = cv2.resize(
            frame_image, None, fx=1 / RESIZE_RATIO, fy=1 / RESIZE_RATIO)
        gray_image = cv2.cvtColor(frame_resize, cv2.COLOR_BGR2GRAY)
        original_landmark, flag = detector.get_landmark(gray_image)
        if not flag:
            continue

        # draw landmark's shape on black image
        original_landmark = detector.resize_landmark(original_landmark,
                                                     RESIZE_RATIO)
        black_image = np.zeros(frame_image.shape, np.uint8)
        black_image = detector.draw_landmark(black_image, original_landmark)

        # cut face from image
        face_rectangle = detector.get_face_rectangle(original_landmark)
        cut_black_image = detector.cut_image(black_image, face_rectangle)
        cut_frame_image = detector.cut_image(frame_image, face_rectangle)
        cut_black_image = cv2.resize(
            cut_black_image, (CROP_SIZE, CROP_SIZE),
            interpolation=cv2.INTER_AREA)
        cut_frame_image = cv2.resize(
            cut_frame_image, (CROP_SIZE, CROP_SIZE),
            interpolation=cv2.INTER_AREA)
        combined_image = np.concatenate(
            [cut_black_image, cut_frame_image], axis=1)
        rgb_image = cv2.cvtColor(
            combined_image,
            cv2.COLOR_BGR2RGB)  # OpenCV uses BGR instead of RGB

        # save image data
        rgb_images.append(rgb_image)
        black_images.append(black_image)
        frame_images.append(frame_image)
        face_rectangles.append(face_rectangle)
        original_landmarks.append(original_landmark)

        count += 1
        if count == args.output_video_fps * args.output_video_time:
            break
    cap.release()
    return rgb_images, black_images, frame_images, face_rectangles, original_landmarks


def save_video(args, display_images):
    video_coder = cv2.VideoWriter_fourcc(*args.output_video_coder)
    video_writer = cv2.VideoWriter(args.output_video_file + ".avi",
                                   video_coder, args.output_video_fps,
                                   (CROP_SIZE * 2, CROP_SIZE))
    for display_image in display_images:
        video_writer.write(display_image)
    video_writer.release()


def display_video(display_images):
    for image in display_images:
        cv2.imshow("frame", image)
        cv2.waitKey(10)
    cv2.destroyAllWindows()


def operator_final_video(args, display_images):
    if args.enable_output_video:
        save_video(args, display_images)
        print("video save complete")
    else:
        display_video(display_images)
        print("video show complete")