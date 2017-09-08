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


test = features[:1000]
print test.shape


start = time.time()

D = facedist.mean_dist(test)

print 'D64 D:', D.shape
print 'D64 sys-size:', sys.getsizeof(D)
print 'D64 np nbytes:', D.nbytes

nrow = len(test)
fdense_distances = np.zeros( (nrow, nrow), dtype=np.double)

for ii in range(nrow):
    for jj in range(ii+1, nrow):
        nn = ii+jj*(jj-1)/2
        rd = D[nn]

        fdense_distances[ii, jj] = rd
        fdense_distances[jj, ii] = rd


print 'Dense64:', fdense_distances.shape
print 'Dense64 sys-size:', sys.getsizeof(fdense_distances)
print 'Dense64 np nbytes:', fdense_distances.nbytes
#print fdense_distances
#print

db = hdbscan.HDBSCAN(min_cluster_size=2, metric='precomputed', core_dist_n_jobs=-1).fit(fdense_distances)
labels = db.labels_
probabilities = db.probabilities_
pers = db.cluster_persistence_

funique, fcounts = np.unique(labels, return_counts=True)

processTime = time.time() - start
print 'facedist distance computation took', processTime, 'found', len(funique)




def split(a, n):
    k, m = divmod(len(a), n)
    return [a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in xrange(n)]


# split features into 4 parts 
# create indices list for parts
indices = split(range(len(test)), 4)

print 'splitted indices:'
for l in indices:
    print len(l), l

start = time.time()


# create threaded, or process pool for calculation
def comp_dist(index):
    return facedistsplit32.mean_dist(test, np.array(index))



executor = ThreadPoolExecutor(max_workers=4)

result = []

for r in executor.map(comp_dist, indices):
    result.append(r)
    #print r
    #print


print 'result:', len(result)

# merge resuls back with hstack
D = np.hstack(result)

print 'SPLIT D32 D:', D.shape
print 'SPLIT D32 sys-size:', sys.getsizeof(D)
print 'SPLIT D32 np nbytes:', D.nbytes
#print D
#print

nrow = len(test)
dense_distances = np.zeros( (nrow, nrow), dtype=np.float32)

for ii in range(nrow):
    for jj in range(ii+1, nrow):
        rd = D[jj, ii]

        dense_distances[ii, jj] = rd
        dense_distances[jj, ii] = rd

print 'SPLIT Dense32:', dense_distances.shape
print 'SPLIT Dense32 sys-size:', sys.getsizeof(dense_distances)
print 'SPLIT Dense32 np nbytes:', dense_distances.nbytes
#print dense_distances
#print


db = hdbscan.HDBSCAN(min_cluster_size=2, metric='precomputed', core_dist_n_jobs=-1).fit(dense_distances)
labels = db.labels_
probabilities = db.probabilities_
pers = db.cluster_persistence_

eunique, ecounts = np.unique(labels, return_counts=True)


processTime = time.time() - start
print 'facedist distance computation took', processTime, 'found', len(eunique)


print fcounts
print ecounts


print np.array_equal(fdense_distances.astype(np.float32), dense_distances)
print np.allclose(fdense_distances.astype(np.float32), dense_distances)
