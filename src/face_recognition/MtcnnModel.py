# -*- coding: utf-8 -*-

'''
Model file module, so that model files are only loaded once when imported
'''

import os
import sys
import tensorflow as tf

from facenet.src import facenet
from facenet.src.align import detect_face


fileDir = os.path.dirname(os.path.realpath(__file__))

facenetDir = os.path.join(fileDir, 'facenet')
facenetModelDir = os.path.join(facenetDir, 'src', 'align',)


session = None
graph = None

# Actual models used for face detection
pnet = None
rnet = None
onet = None


graph = tf.Graph()
session = tf.Session(graph=graph) #config=tf.ConfigProto(inter_op_parallelism_threads=24, intra_op_parallelism_threads=24))
with graph.as_default():
    with session.as_default():
        pnet, rnet, onet = detect_face.create_mtcnn(session, facenetModelDir)
graph.finalize()
