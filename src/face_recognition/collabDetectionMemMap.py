# -*- coding: utf-8 -*-

'''
Detects collaborations of actors from features in db,
split into multiple steps due to memory usage

# Method:
# Measure distances of feature pairs, using mean of embeddings-distance, see facedist.pyx
# Cluster resulting distance matrix with HDBSCAN
# Write found connections into DB

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

import networkx as nx

fileDir = os.path.dirname(os.path.realpath(__file__))


def extend_df(ft, label, proba, pers):
    db = YTDatabase()

    ch = []
    persistents = []

    with db._session_scope(False) as session:

        for i, l in enumerate(label):
            vid = ft['videoID'].iloc[i]
            cid = session.query(Video.channelID).filter(Video.id==vid).first()[0]
            ch.append(cid)
            if int(l) != -1:
                persistents.append(pers[int(l)])
            else:
                persistents.append(0)


    ft['channel'] = np.array(ch)

    ft['label'] = np.array(label)
    ft['proba'] = np.array(proba)
    ft['pers'] = np.array(persistents)


    #fto = ft.iloc[:,[0,1,3,4,5,6,7]]
    #fto = ft.iloc[:,['id', 'videoID', 'duration', 'channel']]

    ftos = ft.sort_values(['channel', 'videoID'])

    return ftos



db = YTDatabase()

start = time.time()

with db._session_scope(False) as session:
    #features = session.query(VideoFeatures).all()
    ft = pa.read_sql(session.query(VideoFeatures.id, VideoFeatures.videoID, VideoFeatures.duration).filter(VideoFeatures.duration > 30.0).statement, db.engine)

#ft = ft[ft.duration > 28.0]
#ft['feature'] = ft['feature'].apply(cp.loads)

processTime = time.time() - start
print 'data extraction took', processTime, 'for', ft.count()


start = time.time()


nrow = len(ft)

dense_distances = np.zeros( (nrow, nrow), dtype=np.double)

sparse_distances = np.load('distance_sparse_30sec.npy', mmap_mode='r')

for ii in range(nrow):
    for jj in range(ii+1, nrow):
        nn = ii+jj*(jj-1)/2
        rd = sparse_distances[nn]

        dense_distances[ii, jj] = rd
        dense_distances[jj, ii] = rd

del sparse_distances

np.save('distances_dense_30sec', dense_distances)

# Max split computation into two executions of this file
del dense_distances

# Try and load dense matrix as memmap, so its not fully loaded into memory for next steps
dense_distances = np.load('distances_dense_30sec.npy', mmap_mode='r')


print dense_distances.flags.owndata

hdb = hdbscan.HDBSCAN(min_cluster_size=2, metric='precomputed', core_dist_n_jobs=-1, gen_min_span_tree=False, memory=fileDir).fit(dense_distances)
hdb_mean_labels = hdb.labels_
hdb_mean_proba = hdb.probabilities_
hdb_mean_pers = hdb.cluster_persistence_

unique, counts = np.unique(hdb_mean_labels, return_counts=True)

processTime = time.time() - start
print 'distance computation took', processTime, 'found', counts

del dense_distances

#ftos = extend_df(ft, hdb_mean_labels, hdb_mean_proba, hdb_mean_pers)

#ftos.to_csv(r'hdb_collab_prefiltered_cluster.txt', sep=str('\t'), encoding='utf-8')


# Filter found collabs for persistence and probability
#ftos = ftos[ftos.pers > 0.05]
#ftos = ftos[ftos.proba > 0.1]

#ftos.to_csv(r'hdb_collab_cluster.txt', sep=str('\t'), encoding='utf-8')


# write cluster-feature id pairs into database
with db._session_scope(True) as session:
    for row in ftos.itertuples():
        session.add(VideoFaceCluster(featureID=row.id, cluster=row.label ))

