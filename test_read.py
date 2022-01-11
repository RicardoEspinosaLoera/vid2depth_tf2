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

tf.compat.v1.disable_eager_execution()
gfile = tf.io.gfile

NUM_SCALES = 4

reader = reader.DataReader("/workspace/vid2depth/vid2depth_tf2/data", 4,128, 416, 3, NUM_SCALES)
image_stack, intrinsic_mat, intrinsic_mat_inv,file_lists = reader.read_data()
print(image_stack.shape)
print(intrinsic_mat.shape)
print(intrinsic_mat_inv.shape)

for a in file_lists['image_file_list']:
    if not gfile.exists(a):
        print(a)
