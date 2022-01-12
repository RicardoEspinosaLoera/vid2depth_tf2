# Copyright 2017 The TensorFlow Authors All Rights Reserved.
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

"""Build model for inference or training."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging
import nets
from ops import icp_grad  # pylint: disable=unused-import
from ops.icp_op import icp
import project
import reader
import tensorflow as tf
import util
import tf_slim as slim
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
import random
import cv2
import numpy as np

tf.compat.v1.disable_eager_execution()
gfile = tf.io.gfile

NUM_SCALES = 4

steps_per_epoch = 0
batch_size = 4
samples = 0

def compile_file_list(data_dir, split, load_pose=False):
    with gfile.GFile(os.path.join(data_dir, '%s.txt' % split), 'r') as f:
      frames = f.readlines()
      subfolders = [x.split(' ')[0] for x in frames]
      frame_ids = [x.split(' ')[1][:-1] for x in frames]
      image_file_list = [
          os.path.join(data_dir, subfolders[i], frame_ids[i] + '.jpg')
          for i in range(len(frames))
      ]
      cam_file_list = [
          os.path.join(data_dir, subfolders[i], frame_ids[i] + '_cam.txt')
          for i in range(len(frames))
      ]    
      
      file_lists = {}
      file_lists['image_file_list'] = image_file_list
      file_lists['cam_file_list'] = cam_file_list
      if load_pose:
        pose_file_list = [
            os.path.join(data_dir, subfolders[i], frame_ids[i] + '_pose.txt')
            for i in range(len(frames))
        ]
        file_lists['pose_file_list'] = pose_file_list
      steps_per_epoch = len(image_file_list) // batch_size
    return file_lists

def unpack_images(image_seq):
    """[h, w * seq_length, 3] -> [h, w, 3 * seq_length]."""
    print("unpack_images "+str(image_seq.shape))
    image_list = [
        image_seq[:, i * 416:(i + 1) * 416, :]
        for i in range(3)
    ]
    print("unpack_images "+str(image_list[0].shape))
    print("unpack_images "+str(image_list[1].shape))
    print("unpack_images "+str(image_list[2].shape))
    image_stack = tf.concat(image_list, axis=2)
    image_stack.set_shape(
        [128, 416, 3 * 3])
    return image_stack

seed = random.randint(0, 2**31 - 1)
file_lists = compile_file_list("/workspace/vid2depth/vid2depth_tf2/data", 'train')
for a in file_lists:
  print(a)
  img = cv2.imread(a)
  print(img.shape)
"""
image_paths_queue = tf.compat.v1.train.string_input_producer(file_lists['image_file_list'], seed=seed, shuffle=True)
#cam_paths_queue = tf.data.TextLineDataset(file_lists['cam_file_list'])
img_reader = tf.compat.v1.WholeFileReader()
_, image_contents = img_reader.read(image_paths_queue)
image_seq = tf.image.decode_image(image_contents)
#print("\nimage_seq "+str(image_seq.shape))
image_stack = unpack_images(image_seq)

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True

epochs = 3

with tf.compat.v1.Session() as sess:
    sess.run([tf.compat.v1.global_variables_initializer(),
        tf.compat.v1.local_variables_initializer()])
    coord = tf.train.Coordinator()
    threads = tf.compat.v1.train.start_queue_runners(coord=coord)
    for _ in range(epochs):
        try:
            while not coord.should_stop():
                print(str(image_stack.shape))
                sess.run(image_stack)
                print(str(image_stack.shape))
                samples += batch_size
                print(samples, "samples have been seen")
        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:
            coord.request_stop()
    coord.join(threads)
    """