# -*- coding: utf-8 -*-

'''
Due to memory usage problems, if features array is present as file on-disk, 
its loaded here and used for computing sparse distance matrix.

Features array cant be loaded as numpy memmap, as its not a "perfect" array -> every row has a different length.

'''


from __future__ import unicode_literals

from concurrent.futures import * 

import os
import time
import numpy as np
import pandas as pa
import cPickle as cp
import json

import math

from threading import Thread

from database import *

from scipy.spatial.distance import cdist, pdist, squareform
from scipy.stats import describe

import itertools
import string

import hdbscan
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering

import facedist32

import networkx as nx

fileDir = os.path.dirname(os.path.realpath(__file__))

# Load features array from disk
features = np.load(os.path.join(fileDir,'features_3MONTH.npy'))

print 'Loaded feature:', features.shape

np.save('features_3MONTH_15.npy', np.asarray([f[:15] for f in features]))
