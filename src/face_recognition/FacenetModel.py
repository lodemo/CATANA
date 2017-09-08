# -*- coding: utf-8 -*-

'''
Model file module, so that model files are only loaded once when imported
'''

import os
import sys
import tensorflow as tf

from facenet.src import facenet

fileDir = os.path.dirname(os.path.realpath(__file__))

facenetDir = os.path.join(fileDir, 'facenet')
#facenetModelDir = os.path.join(facenetDir, 'models', '20161116-234200')
#facenetModelDir = os.path.join(facenetDir, 'models', '20170117-215115')
#facenetModelDir = os.path.join(facenetDir, 'models', '20170216-091149')
#frozen_graph_filename = os.path.join(facenetDir, 'models', '20161116-234200.pb')
frozen_graph_filename = os.path.join(facenetDir, 'models', '20170117-215115.pb')
#frozen_graph_filename = os.path.join(facenetDir, 'models', '20170216-091149.pb2')


def load_graph(frozen_graph_filename):
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def,
            input_map=None,
            return_elements=None,
            op_dict=None,
            name='resnet',
            producer_op_list=None
        )

    return graph


session = None
graph = None

#graph = tf.Graph()
graph = load_graph(frozen_graph_filename)
session = tf.Session(graph=graph) #config=tf.ConfigProto(inter_op_parallelism_threads=24, intra_op_parallelism_threads=24))
with graph.as_default():
    with session.as_default():
        # Load the model
        pass
        #meta_file, ckpt_file = facenet.get_model_filenames(facenetModelDir)
        #facenet.load_model(facenetModelDir, meta_file, ckpt_file)
graph.finalize()
