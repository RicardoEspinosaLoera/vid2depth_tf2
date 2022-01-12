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

tf.compat.v1.disable_eager_execution()
gfile = tf.io.gfile

NUM_SCALES = 4

"""
reader = reader.DataReader("/workspace/vid2depth/vid2depth_tf2/data", 4,128, 416, 3, NUM_SCALES)
image_stack, intrinsic_mat, intrinsic_mat_inv,image_contents = reader.read_data()
print(image_stack.shape)
print(intrinsic_mat.shape)
print(str(image_contents))
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
img_batch = fname_data.map(lambda fname: \
        tf.image.decode_image(tf.read_file(fname),3))\
        .batch(batch_sz).make_initializable_iterator()

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