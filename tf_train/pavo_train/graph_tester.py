

import numpy as np
import os
import six.moves.urllib as urllib
import sys, getopt
import tarfile
import tensorflow as tf


from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image


from utils import label_map_util
from utils import visualization_utils as vis_util

import cv2
import time

#######################################################
#                     NOTES                           #
#######################################################




path_to_graph = None
path_to_photos = None
path_to_labels = None


# Print helper mesage
def usage():
  sys.exit('Error specifying input or output folder \n' \
    + 'custom_resize.py -g <graph_path_folder> -f <photos_path_folder>')

NUM_ARGS = 3
# Get paths to test photos and gaph
def retrieve_paths(argv):
  global path_to_photos
  global path_to_graph
  global path_to_labels

  # Verify params
  try:
    if (not (len(argv)>NUM_ARGS)):
      usage()

    opts, args = getopt.getopt(argv,"hl:g:f:",["labels_folder=","graph_folder=","photos_folder="])
  except getopt.GetoptError:
    usage()

  # Get folder names
  for opt, arg in opts:
    if opt == '-h':
      usage()
    elif opt in ('-g', "--graph_folder"):
      # get the .pb inside the folder
      for file in os.listdir(arg):
        if file.endswith(".pb"):
          path_to_graph = arg + '/' + file
    elif opt in ("-f", "--photos_folder"):
      path_to_photos = arg
    elif opt in ("-l", "--labels_folder"):
      # get the .pbtxt inside the folder
      for file in os.listdir(arg):
        if file.endswith(".pbtxt"):
          path_to_labels = arg + '/' + file

  print "File paths are:"
  print "Graph : ", path_to_graph
  print "Labels : ", path_to_labels
  print "Photos : ", path_to_photos

  if (path_to_photos == None or path_to_graph == None or path_to_photos == None):
    raise  ValueError('Paths are incomplete or misgiven')





def main(argv):
  retrieve_paths(argv)

  # Total number of classes detected by graph
  NUM_CLASSES = 1

  # Load label mappings
  label_map = label_map_util.load_labelmap(path_to_labels)
  categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
  class_index_mapping = label_map_util.create_category_index(categories)

  # Loading detection grapho
  detection_graph = tf.Graph()
  with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(path_to_graph, 'rb') as fid:
      serialized_graph = fid.read()
      od_graph_def.ParseFromString(serialized_graph)
      tf.import_graph_def(od_graph_def, name='')

  # Running TF session
  with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:

      list_of_images_paths = os.listdir(path_to_photos)

      for img_path in list_of_images_paths:

        if not img_path.endswith(".jpg"):
          continue
        #######################################################
        #      CHANGE TO RETRIEVE IMAGE FROM ROS NODE         #
        #######################################################
        current_image_np = cv2.imread(path_to_photos + img_path, cv2.IMREAD_COLOR)
        cv2.imshow('image',current_image_np)
        # cv2.waitKey(5000)
        ## PROCESS DETECTION ##
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        current_image_np_expanded = np.expand_dims(current_image_np, axis=0)
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        scores = detection_graph.get_tensor_by_name('detection_scores:0')
        classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        
        # Get time per frame
        start_time = time.time()
        
        # Run session for detecion
        (boxes, scores, classes, num_detections) = sess.run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor: current_image_np_expanded})
        end_time = time.time()


        print("Elapsed Time per Frame:", end_time-start_time)


        # Visualization of the results of a detection.
        # Mapping to labels is on 'class index mapping'
        # min_score_thresh default is 0.5
        vis_util.visualize_boxes_and_labels_on_image_array(
            current_image_np,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            class_index_mapping,
            use_normalized_coordinates=True,
            line_thickness=8,
            min_score_thresh=0.5)

        # Show image and wait for break
        # Resize to a convenient size for displaying
        # cv2.imshow('image',cv2.resize(current_image_np,(1280,960)))
        cv2.imshow('image',current_image_np)
        if cv2.waitKey(3000) & 0xFF == ord('q'):
			cv2.destroyAllWindows()
			break


if __name__ == '__main__':
  main(sys.argv[1:])

