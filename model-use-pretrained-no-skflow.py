# Kleban's Notes
#
# This file uses the new model and performs the predictions.
#
#
# This is a modified version of a file from the tensorflow repo.
# Original file found here:
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/image/imagenet/classify_image.py
#
#
# This file is mostly left unchanged, accept for:
#
# - Needed to change the prediction logic to test multiple files, not a single file
# - Added the support for caching the test image data set
# - Outputs two different prediction submission files.
#
#
# ==============================================================================

# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================



"""Simple image classification with Inception.

Run image classification with Inception trained on ImageNet 2012 Challenge data
set.

This program creates a graph from a saved GraphDef protocol buffer,
and runs inference on an input JPEG image. It outputs human readable
strings of the top 5 predictions along with their probabilities.

Change the --image_file argument to any jpg image to compute a
classification of that image.

Please see the tutorial and website for a detailed description of how
to use this script to perform image recognition.

https://tensorflow.org/tutorials/image_recognition/
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import re
import sys
import tarfile
import glob
import pandas as pd
import datetime
import pickle

import numpy as np
from six.moves import urllib
import tensorflow as tf

from tensorflow.python.platform import gfile
from tensorflow.python.client import graph_util
from tensorflow.python.framework import tensor_shape



# Kleban's Vars
pbfilename = 'output_graph_june1-test.pb'
pblabelfilename='output_labels_june1-test.txt' # was 'imagenet_2012_challenge_label_map_proto.pbtxt'
uidfilename='output_labels_june1-test.txt' # was 'imagenet_synset_to_human_label_map.txt'

pathoftestimages='/data/new-images/input/test/'

imagestoprocess=100000
use_input_cache=False #Change this to false if you want to re-process the images with a new preprocessing settings (like # of pixels, etc)
input_cache_file='objs-transferlearning-testdata.pickle'

FLAGS = tf.app.flags.FLAGS


# classify_image_graph_def.pb:
#   Binary representation of the GraphDef protocol buffer.
# imagenet_synset_to_human_label_map.txt:
#   Map from synset ID to a human readable string.
# imagenet_2012_challenge_label_map_proto.pbtxt:
#   Text representation of a protocol buffer mapping a label to synset ID.
tf.app.flags.DEFINE_string(
    'model_dir', '/tmp/imagenet',
    """Path to classify_image_graph_def.pb, """
    """imagenet_synset_to_human_label_map.txt, and """
    """imagenet_2012_challenge_label_map_proto.pbtxt.""")

#tf.app.flags.DEFINE_string('image_file', '',
#                           """Absolute path to image file.""")
#tf.app.flags.DEFINE_integer('num_top_predictions', 5,
#                            """Display this many predictions.""")

# pylint: disable=line-too-long
DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
# pylint: enable=line-too-long


def create_graph():
  """Creates a graph from saved GraphDef file and returns a saver."""
  # Creates graph from saved graph_def.pb.
  with tf.gfile.FastGFile(os.path.join(
      FLAGS.model_dir, pbfilename), 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

## try reading all files first, then do predictions
## trying this to see if I can improve the speed.

def run_inference_on_images_serial(pathoftestimages):

  testing_images = create_image_lists(pathoftestimages)
  print('Total files to test:')
  print(len(testing_images))
  allpredictions=[]
  allfiles=[]
  images=[]



  #### Read files

  imageslooped=0



  if use_input_cache == False:
      print ('reading images from FILES...')
      for file in testing_images:
        print(file , ' - ' , imageslooped)
        filewithpath=pathoftestimages + file
        image_data = tf.gfile.FastGFile(filewithpath, 'rb').read()
        images.append(image_data)
        allfiles.append(file)
        imageslooped=imageslooped+1
        if imageslooped > imagestoprocess:
            break

      # Saving the objects:
      with open(input_cache_file, 'w') as f:
          pickle.dump([images,allfiles], f)

  if use_input_cache == True:
      print('Loading Inputs from CACHE')
      with open(input_cache_file) as f:
          images,allfiles = pickle.load(f)




  #if not tf.gfile.Exists(image):
    # tf.logging.fatal('File does not exist %s', image)

  # Creates graph from saved GraphDef.
  create_graph()






  with tf.Session() as sess:
    # Some useful tensors:
    # 'softmax:0': A tensor containing the normalized prediction across
    #   1000 labels.
    # 'pool_3:0': A tensor containing the next-to-last layer containing 2048
    #   float description of the image.
    # 'DecodeJpeg/contents:0': A tensor containing a string providing JPEG
    #   encoding of the image.
    # Runs the softmax tensor by feeding the image_data as input to the graph.

    ## kleban: Switch this up to match new transfer learning last layer.
    #softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

    print('making predictions...')
    imageslooped=0
    for image in images:

        #print(file , ' - ' , imageslooped)
        #filewithpath=pathoftestimages + file
        #image_data = tf.gfile.FastGFile(filewithpath, 'rb').read()
        #run_inference_on_image(filewithpath)
        predictions = sess.run(softmax_tensor,{'DecodeJpeg/contents:0': image})
        #print(predictions)
        predictions = np.squeeze(predictions)
        #print(predictions)

        allpredictions.append(predictions)
        #allfiles.append(file)
        imageslooped=imageslooped+1
        if imageslooped > imagestoprocess:
            break


    return allpredictions,allfiles






def maybe_download_and_extract():
  """Download and extract model tar file."""
  dest_directory = FLAGS.model_dir
  if not os.path.exists(dest_directory):
    os.makedirs(dest_directory)
  filename = DATA_URL.split('/')[-1]
  filepath = os.path.join(dest_directory, filename)
  if not os.path.exists(filepath):
    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' % (
          filename, float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()
    filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
  tarfile.open(filepath, 'r:gz').extractall(dest_directory)



def create_image_lists(image_dir):
  """Builds a list of training images from the file system.

  Analyzes the sub folders in the image directory, splits them into stable
  training, testing, and validation sets, and returns a data structure
  describing the lists of images for each label and their paths.

  Args:
    image_dir: String path to a folder containing subfolders of images.
    testing_percentage: Integer percentage of the images to reserve for tests.
    validation_percentage: Integer percentage of images reserved for validation.

  Returns:
    A dictionary containing an entry for each label subfolder, with images split
    into training, testing, and validation sets within each label.
  """
  if not gfile.Exists(image_dir):
    print("Image directory '" + image_dir + "' not found.")
    return None
  result = {}
  sub_dirs = [x[0] for x in os.walk(image_dir)]
  # The root directory comes first, so skip it.
  is_root_dir = True
  for sub_dir in sub_dirs:
    print('in sub loop')
    extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
    file_list = []
    dir_name = os.path.basename(image_dir)
    print("Looking for images in '" + image_dir + "'")
    for extension in extensions:
      file_glob = os.path.join(image_dir, dir_name, '*.' + extension)
      file_list.extend(glob.glob(file_glob))
    if not file_list:
      print('No files found')
      continue
    if len(file_list) < 20:
      print('WARNING: Folder has less than 20 images, which may cause issues.')
    label_name = re.sub(r'[^a-z0-9]+', ' ', dir_name.lower())
    testing_images = []
    for file_name in file_list:
      base_name = os.path.basename(file_name)
      # We want to ignore anything after '_nohash_' in the file name when
      # deciding which set to put an image in, the data set creator has a way of
      # grouping photos that are close variations of each other. For example
      # this is used in the plant disease data set to group multiple pictures of
      # the same leaf.
      hash_name = re.sub(r'_nohash_.*$', '', file_name)
      # This looks a bit magical, but we need to decide whether this file should
      # go into the training, testing, or validation sets, and we want to keep
      # existing files in the same set even if more files are subsequently
      # added.
      # To do that, we need a stable way of deciding based on just the file name
      # itself, so we do a hash of that and then use that to generate a
      # probability value that we use to assign it.
      testing_images.append(base_name)
  return testing_images



def create_submission(predictions, test_id, info):
    result1 = pd.DataFrame(predictions, columns=['c9', 'c8', 'c3', 'c2', 'c1', 'c0', 'c7', 'c6', 'c5', 'c4'])
    result1.loc[:, 'img'] = pd.Series(test_id, index=result1.index)
    now = datetime.datetime.now()
    if not os.path.isdir('subm'):
        os.mkdir('subm')
    suffix = info + '_' + str(now.strftime("%Y-%m-%d-%H-%M"))
    sub_file = os.path.join('subm', 'submission_' + suffix + '.csv')
    result1.to_csv(sub_file, index=False)


def main(_):
  maybe_download_and_extract()
  allpredictions,allfiles=run_inference_on_images_serial(pathoftestimages)

  ## Write out submission File - Use exact values from model
  info_string="transfer-from-inception3"
  create_submission(allpredictions, allfiles, info_string)

  ## Write out submission File - Try modifying the results a bit to take less risk.
  predictions_for_test_changed = np.array(allpredictions)
  predictions_for_test_changed = predictions_for_test_changed/1.2
  predictions_for_test_changed = predictions_for_test_changed+0.1
  info_string="transfer-from-inception3-with-0.1-additions"
  create_submission(predictions_for_test_changed, allfiles, info_string)




if __name__ == '__main__':
  tf.app.run()
