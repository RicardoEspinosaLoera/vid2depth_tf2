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

tf.compat.v1.disable_eager_execution()
gfile = tf.io.gfile

NUM_SCALES = 4


steps_per_epoch = 0
batch_size = 4

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

seed = random.randint(0, 2**31 - 1)
file_lists = compile_file_list("/workspace/vid2depth/vid2depth_tf2/data", 'train')
image_paths_queue = tf.compat.v1.train.string_input_producer(file_lists['image_file_list'], seed=seed, shuffle=True)
#cam_paths_queue = tf.compat.v1.train.string_input_producer(file_lists['cam_file_list'], seed=seed, shuffle=True)
#cam_paths_queue = tf.data.TextLineDataset(self.file_lists['cam_file_list'])
img_reader = tf.compat.v1.WholeFileReader()
_, image_contents = img_reader.read(image_paths_queue)
image_seq = tf.image.decode_image(image_contents)

with tf.Session() as sess:
    sess.run([tf.global_variables_initializer(),
        tf.local_variables_initializer()])
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    for _ in range(epochs):
        try:
            while not coord.should_stop():
                sess.run(image_seq)
                samples += batch_sz;
                print(samples, "samples have been seen")
        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:
            coord.request_stop();
    coord.join(threads)
"""
reader = reader.DataReader("/workspace/vid2depth/vid2depth_tf2/data", 4,128, 416, 3, NUM_SCALES)
image_stack, intrinsic_mat, intrinsic_mat_inv,image_contents = reader.read_data()
print(image_stack.shape)
print(intrinsic_mat.shape)
print(str(image_contents))
"""

"""
if not os.path.exists('example'):
    shutil.rmTree('example')
    os.mkdir('example')

batch_sz = 10; epochs = 2; buffer_sz = 30; samples = 0
for i in range(50):
    _x = np.random.randint(0, 256, (10, 10, 3), np.uint8)
    plt.imsave("example/image_{}.jpg".format(i), _x)
fname_data = tf.data.Dataset.list_files('example/*.jpg')\
        .shuffle(buffer_sz).repeat(epochs)
img_batch = fname_data.map(lambda fname: tf.image.decode_image(tf.io.read_file(fname),3)).batch(batch_sz).make_initializable_iterator()

with tf.Session() as sess:
    sess.run([img_batch.initializer,
        tf.global_variables_initializer(),
        tf.local_variables_initializer()])
    next_element = img_batch.get_next()
    try:
        while True:
            sess.run(next_element)
            samples += batch_sz
            print(samples, "samples have been seen")
    except tf.errors.OutOfRangeError:
        pass
    print('Done training -- epoch limit reached')
"""

