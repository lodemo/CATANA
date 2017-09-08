# -*- coding: utf-8 -*-

'''
Due to memory usage problems, if features array is present as file on-disk, 
its loaded here and used for computing sparse distance matrix.

Features array cant be loaded as numpy memmap, as its not a "perfect" array -> every row has a different length.

'''


from __future__ import unicode_literals

from concurrent.futures import * 

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
import facedist32

import networkx as nx

fileDir = os.path.dirname(os.path.realpath(__file__))

# Load features array from disk
features = np.load(os.path.join(fileDir,'features_30sec_fixed.npy'))

print 'Loaded feature:', features.shape


test = features[:10000]
print test.shape


start = time.time()

D = facedist.mean_dist(test)

print 'D64 sys-size:', sys.getsizeof(D)
print 'D64 np nbytes:', D.nbytes

nrow = len(test)
dense_distances = np.zeros( (nrow, nrow), dtype=np.double)

for ii in range(nrow):
    for jj in range(ii+1, nrow):
        nn = ii+jj*(jj-1)/2
        rd = D[nn]

        dense_distances[ii, jj] = rd
        dense_distances[jj, ii] = rd


print 'Dense64 sys-size:', sys.getsizeof(dense_distances)
print 'Dense64 np nbytes:', dense_distances.nbytes


db = hdbscan.HDBSCAN(min_cluster_size=2, metric='precomputed', core_dist_n_jobs=-1).fit(dense_distances)
labels = db.labels_
probabilities = db.probabilities_
pers = db.cluster_persistence_

funique, fcounts = np.unique(labels, return_counts=True)


processTime = time.time() - start
print 'facedist distance computation took', processTime, 'found', len(funique)



start = time.time()

D = facedist32.mean_dist(test)

print 'D32 sys-size:', sys.getsizeof(D)
print 'D32 np nbytes:', D.nbytes


nrow = len(test)
dense_distances = np.zeros( (nrow, nrow), dtype=np.float32)

for ii in range(nrow):
    for jj in range(ii+1, nrow):
        nn = ii+jj*(jj-1)/2
        rd = D[nn]

        dense_distances[ii, jj] = rd
        dense_distances[jj, ii] = rd

print 'Dense32 sys-size:', sys.getsizeof(dense_distances)
print 'Dense32 np nbytes:', dense_distances.nbytes


db = hdbscan.HDBSCAN(min_cluster_size=2, metric='precomputed', core_dist_n_jobs=-1).fit(dense_distances)
labels = db.labels_
probabilities = db.probabilities_
pers = db.cluster_persistence_

eunique, ecounts = np.unique(labels, return_counts=True)


processTime = time.time() - start
print 'facedist distance computation took', processTime, 'found', len(eunique)



print fcounts
print ecounts


