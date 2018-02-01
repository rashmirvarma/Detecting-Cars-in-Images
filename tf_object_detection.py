'''

TensorFlow Object Detection Program

'''

try:
    import numpy as np
    import os
    from collections import defaultdict
    from matplotlib import pyplot as plt
    from PIL import Image
except:
    ImportError
try:
    import six.moves.urllib as urllib
except:
    ImportError
    print "Please install six.moves on the system"
try:
    import sys
    work_dir = '/Users/rashmivarma/tensorflow/python/models/research/'
    sys.path.append(work_dir)
except:
    ImportError
    print "Failure while importing the sys library"
try:
    import tarfile
except:
    ImportError
    print "Please install the tarfile library"
try:
    import tensorflow as tf
except:
    ImportError
    print "Please check installation guide to install tensorflow"
try:
    import zipfile
except:
    ImportError
    print "Please install zipfile library"
try:
    from io import StringIO
except:
    ImportError
    print "Please install the io library"
try:
    from object_detection.utils import label_map_util
    from object_detection.utils import visualization_utils as vis_util
except:
    print "Error while importing label_map_util or visualization_utils "
    print "Please check if importing is being done correctly"

# What model to download.
MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join(
    work_dir, 'object_detection/data', 'mscoco_label_map.pbtxt')

# Number of classes to detect is set to 1 as it only needs to detect cars
NUM_CLASSES = 1

# Downloading the model file
opener = urllib.request.URLopener()
opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
tar_file = tarfile.open(MODEL_FILE)
for file in tar_file.getmembers():
    file_name = os.path.basename(file.name)
    if 'frozen_inference_graph.pb' in file_name:
        tar_file.extract(file, os.getcwd())

# Reading the tensorflow graph
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


# To test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
PATH_TO_TEST_IMAGES_DIR = '/Users/rashmivarma/PycharmProjects/256Proj/test_images 1/'
fileList = os.listdir(PATH_TO_TEST_IMAGES_DIR)
TEST_IMAGE_PATHS = [os.path.join(PATH_TO_TEST_IMAGES_DIR, fileList[i]) for i in range(1, len(fileList))]

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)


with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        # Definite input and output Tensors for detection_graph
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        for image_path in TEST_IMAGE_PATHS:
            image = Image.open(image_path)
            # the array based representation of the image will be used later in order to prepare the
            # result image with boxes and labels on it.
            image_np = load_image_into_numpy_array(image)
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            # Actual detection.
            (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
            # Visualization of the results of a detection.
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=8)

            # Plotting the figure in the above defined image sizes
            plt.figure(1,figsize=IMAGE_SIZE)
            plt.imshow(image_np)
            plt.show()
