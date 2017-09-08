# -*- coding: utf-8 -*-

'''
Detects collaborations of actors from features in db

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


def hdbscan_tests(features, ftype='mean', min_cluster_size=2):
    
    if ftype=='min':
        D = facedist.min_dist(features)
    elif ftype=='max':
        D = facedist.max_dist(features)
    elif ftype=='meanmin':
        D = facedist.meanmin_dist(features)
    else:
        D = facedist.mean_dist(features)

    #print D.shape

    db = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric='precomputed').fit(D)
    labels = db.labels_
    probabilities = db.probabilities_
    pers = db.cluster_persistence_

    unique, counts = np.unique(labels, return_counts=True)

    return np.asarray((unique, counts)).T, labels, probabilities, pers


def dbscan_tests(features, ftype='mean', eps=0.7, min_samples=2):
    
    if ftype=='min':
        D = facedist.min_dist(features)
    elif ftype=='max':
        D = facedist.max_dist(features)
    elif ftype=='meanmin':
        D = facedist.meanmin_dist(features)
    else:
        D = facedist.mean_dist(features)

    #print D.shape

    db = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed').fit(D)
    labels = db.labels_

    unique, counts = np.unique(labels, return_counts=True)

    return np.asarray((unique, counts)).T, labels, None


def agglo_tests(features, ftype='mean', num_cluster=8, metric='precomputed'):
    
    if ftype=='min':
        D = facedist.min_dist(features)
    elif ftype=='max':
        D = facedist.max_dist(features)
    elif ftype=='meanmin':
        D = facedist.meanmin_dist(features)
    else:
        D = facedist.mean_dist(features)

    #print D.shape

    Hclustering = AgglomerativeClustering(n_clusters=num_cluster, affinity=metric, linkage='average')
    Hclustering.fit(D)
    labels = Hclustering.labels_

    unique, counts = np.unique(labels, return_counts=True)

    return np.asarray((unique, counts)).T, labels, None



def extend_df(ft, label, proba, pers):
    db = YTDatabase()

    ch = []
    persistents = []

    with db._session_scope(False) as session:

        for i, l in enumerate(hdb_mean_labels):
            vid = ft['videoID'].iloc[i]
            cid = session.query(Video.channelID).filter(Video.id==vid).first()[0]
            ch.append(cid)
            if int(l) != -1:
                persistents.append(hdb_mean_pers[int(l)])
            else:
                persistents.append(0)


    ft['channel'] = np.array(ch)

    ft['label'] = np.array(hdb_mean_labels)
    ft['proba'] = np.array(hdb_mean_proba)
    ft['pers'] = np.array(persistents)


    fto = ft.iloc[:,[0,1,3,4,5,6,7]]

    ftos = fto.sort_values(['channel', 'videoID'])

    return ftos



db = YTDatabase()

start = time.time()

with db._session_scope(False) as session:
    #features = session.query(VideoFeatures).all()
    ft = pa.read_sql(session.query(VideoFeatures).filter(VideoFeatures.duration > 30.0).statement, db.engine)

#ft = ft[ft.duration > 30.0]
#ft = ft[ft.duration > 28.0]
ft['feature'] = ft['feature'].apply(cp.loads)

features = ft['feature'].values

processTime = time.time() - start
print 'data extraction took', processTime, 'for', features.shape


start = time.time()

#clustering
uqc, hdb_mean_labels, hdb_mean_proba, hdb_mean_pers = hdbscan_tests(features, ftype='mean', min_cluster_size=2)


processTime = time.time() - start
print 'distance computation took', processTime


ftos = extend_df(ft.copy(), hdb_mean_labels, hdb_mean_proba, hdb_mean_pers)

ftos.to_csv(r'hdb_collab_pre-filterd_cluster.txt', sep=str('\t'), encoding='utf-8')

ftos = ftos[ftos.pers > 0.05]

ftos.to_csv(r'hdb_collab_cluster.txt', sep=str('\t'), encoding='utf-8')


# write cluster-feature id pairs into database
with db._session_scope(True) as session:
    for row in ftos.itertuples():
        session.add(VideoFaceCluster(featureID=row.id, cluster=row.label ))
