from __future__ import unicode_literals
import youtube_dl


import os
import sys

import tensorflow as tf

import FacenetModel


session = FacenetModel.session


with session.graph.as_default():
    with session.as_default():

        file_writer = tf.summary.FileWriter('tf_logs', session.graph)
