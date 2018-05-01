#simple script to load graph and predict boxes
import os
from os import makedirs, path as op
import sys
import glob
import six.moves.urllib as urllib
import tensorflow as tf
import tarfile
import cv2

from io import StringIO
import zipfile
import numpy as np
from collections import defaultdict
from PIL import ImageDraw, Image

sys.path.append("..")

from utils import label_map_util
from utils import visualization_utils as vis_util

flags = tf.app.flags
flags.DEFINE_string('model_name', '', 'Path to frozen detection graph')
flags.DEFINE_string('path_to_label', '', 'Path to label file')
flags.DEFINE_string('test_image_path', '', 'Path to test imgs and output diractory')
flags.DEFINE_string('image_name', '', 'Name of the image to check')
flags.DEFINE_boolean('display_image', False, 'Display the image after detection')
FLAGS = flags.FLAGS

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

def predict_single_image():
    print("Starting session...")

    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    print("Starting started...")
    print("Check path: ", FLAGS.test_image_path)

    while True:
        image_name = input('Enter the image name: ')

        if image_name == 'exit':
            print("shutting down the program...")
            return

        image_path = FLAGS.test_image_path + '/' + image_name

        print("going to open: ", image_path)

        print("loading image...")
        image = Image.open(image_path)
        image_np = load_image_into_numpy_array(image)
        
        image_np_expanded = np.expand_dims(image_np, axis=0)
        
        print("processing in session...")
        (boxes, scores, classes, num) = sess.run(
          [detection_boxes, detection_scores, detection_classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})

        print("visualising hitboxes...")
        vis_image = vis_util.visualize_boxes_and_labels_on_image_array(
                 image_np,
                 np.squeeze(boxes[0]),
                 np.squeeze(classes[0]).astype(np.int32),
                 np.squeeze(scores[0]),
                 category_index,
                 use_normalized_coordinates=True,
                 line_thickness=1)
        print("{} boxes in {} image tile!".format(len(boxes), image_name))

        image_pil = Image.fromarray(np.uint8(vis_image)).convert('RGB')
        cv_im = np.uint8(vis_image)
        cv_im = cv2.cvtColor(cv_im, cv2.COLOR_RGB2BGR)

        if FLAGS.display_image:
            print("displaying image...")
            cv2.imshow("img", cv_im)
            cv2.waitKey()
            cv2.destroyAllWindows()

    # with tf.gfile.Open(image_path, 'w') as fid:
    #      image_pil.save(fid, 'PNG')



if __name__ =='__main__':
    model_name = op.join(os.getcwd(), FLAGS.model_name)
    path_to_ckpt = op.join(model_name,  'frozen_inference_graph.pb')
    path_to_label = op.join(os.getcwd(), FLAGS.path_to_label)
    num_classes = 1
    #test_image_path = op.join(os.getcwd(), FLAGS.test_image_path)
    #check_img_path = glob.glob(test_image_path + "/" + FLAGS.image_name)

    ############
    #Load the frozen tensorflow model
    #############

    print("Loading detection graph...")
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(path_to_ckpt, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    print("Graph loaded")
    ############
    #Load the label file
    #############

    print("Loading label map...")
    label_map = label_map_util.load_labelmap(path_to_label)
    print("Labelmap loaded")
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=num_classes, use_display_name=True)
    print("Graph contains: ", categories)

    category_index = label_map_util.create_category_index(categories)
    print("Index: ", category_index)

    predict_single_image()