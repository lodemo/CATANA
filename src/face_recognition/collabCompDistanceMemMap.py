# -*- coding: utf-8 -*-

'''
Due to memory usage problems, if features array is present as file on-disk, 
its loaded here and used for computing sparse distance matrix.

Features array cant be loaded as numpy memmap, as its not a "perfect" array -> every row has a different length.

'''

# MIT License
# 
# Copyright (c) 2017 Moritz Lode
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

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
features = np.load(os.path.join(fileDir,'features_3MONTH_15.npy'))

print 'Loaded feature:', features.shape

# Compute sparse matrix, see cython_sparse_arr
D = facedist32.mean_dist(features)

# Save sparse matrix on disk for further computation
np.save('distance_sparse_3MONTH_15.npy', D)

'''
del features
# the callable
def generate_array_on_disk(d):
    np.save('distance_sparse.npy', d)
    #np.save('distance_dense.npy', d.todense())

with ProcessPoolExecutor(max_workers=1) as subprocess:
    subprocess.submit(generate_array_on_disk, Df).result()

'''
