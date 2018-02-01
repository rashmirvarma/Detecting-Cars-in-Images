'''
Object Detection in Videos
'''

import os
import six.moves.urllib as urllib
import tarfile
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

import os
import cv2
import time
import argparse
import multiprocessing
import numpy as np
import tensorflow as tf

from matplotlib import pyplot as plt
import sys

sys.path.append('/Users/rashmivarma/tensorflow/python/models/research/')

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

CWD_PATH = "/Users/rashmivarma/tensorflow/python/models/research/"

# Path to frozen detection graph. This is the actual model that is used for the object detection.
MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
PATH_TO_CKPT = os.path.join(CWD_PATH, 'object_detection/models', MODEL_NAME, 'frozen_inference_graph.pb')

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join(CWD_PATH, 'object_detection', 'data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

# Loading label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)


def detect_objects(image_np, sess, detection_graph):
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    # Each box represents a part of the image where a particular object was detected.
    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    # Actual detection.
    (boxes, scores, classes, num_detections) = sess.run(
        [boxes, scores, classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})
    final_classes = np.squeeze(classes)
    if final_classes[0] == 3:
    # Visualization of the results of a detection.
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=8)
    return image_np


# First test on images
PATH_TO_TEST_IMAGES_DIR = '/Users/rashmivarma/tensorflow/python/models/research/object_detection/test_images'
print PATH_TO_TEST_IMAGES_DIR
TEST_IMAGE_PATHS = [os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 3)]

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)



# Load a frozen TF model
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        for image_path in TEST_IMAGE_PATHS:
            image = Image.open(image_path)
            image_np = load_image_into_numpy_array(image)
            image_process = detect_objects(image_np, sess, detection_graph)
            # print(image_process.shape)
            plt.figure(figsize=IMAGE_SIZE)
            plt.imshow(image_process)

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML
import imageio

imageio.plugins.ffmpeg.download()
import moviepy.editor
from moviepy.editor import *
from moviepy.config import get_setting


def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # you should return the final output (image with lines are drawn on lanes)
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            image_process = detect_objects(image, sess, detection_graph)
            return image_process

#Save images to output_main
white_output = 'output_main.mp4'
clip1 = VideoFileClip('main.mp4')


duration = 12
segment_length = 1
clip_start = 0



clips = []
# Merge videos
from moviepy.editor import VideoFileClip, concatenate_videoclips
import moviepy.editor as mp

# Looping through subclips and applying object detection to every subclip
while clip_start != duration:
    clip_end = clip_start + segment_length

    # make sure the the end of the clip doesn't exceed the length of the original video
    if clip_end > duration:
        clip_end = duration

    # create a new moviepy videoclip, and add it to our clips list
    clip = clip1.subclip(clip_start, clip_end)
    clip = clip.fl_image(process_image)
    clips.append(clip)

    clip_start = clip_end

    final_video = mp.concatenate_videoclips(clips)
final_video.write_videofile("output_main.mp4", bitrate="5000k",fps=30)

clip3 = VideoFileClip("output_main.mp4")
clip3.write_gif("mainGif.gif")
