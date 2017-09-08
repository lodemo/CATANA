# -*- coding: utf-8 -*-

'''
 To be able to compute the distance matrix clustered on different machines, this script splits, and creates indices to the data.

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

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

import os
import sys
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

import facedist
import facedistsplit32

import networkx as nx

fileDir = os.path.dirname(os.path.realpath(__file__))

# Load features array from disk
features = np.load(os.path.join(fileDir,'features_3MONTH_15.npy'))

print 'Loaded feature:', features.shape


def split(a, n):
    k, m = divmod(len(a), n)
    return [a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in xrange(n)]


# split features into parts
# create indices list for parts
indices = split(range(len(features)), 6)

print 'splitted indices:'
for l in indices:
    print len(l)

# save indices to disk
np.save('features_INDICES', np.asarray(indices))
