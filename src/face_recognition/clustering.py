# -*- coding: utf-8 -*-

'''
File contains different clustering methods which were tested in evaluation.

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

import numpy as np
import pandas as pa
from scipy.spatial.distance import euclidean, pdist, cdist, squareform
from sklearn.cluster import DBSCAN, Birch

from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial import distance

from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans

import hdbscan



def dbscan_cluster(feature, eps=0.67, min_samples=5, metric='euclidean'):
    
    data = [fi for (ni, nj, fi) in feature]

    if len(data) < min_samples:
        return np.asarray((0,0)), {}

    db = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean').fit(data)
    core_samples_mask = np.zeros_like(db.labels_, dtype=np.float)
    core_samples_mask[db.core_sample_indices_] = 1.0 # Mask core sample with higher "probability"
    labels = db.labels_

    # Number of clusters in labels, INCLUDING NOISE -1
    n_clusters = len(set(labels))

    #print('Estimated number of clusters: %d' % n_clusters)
    #print 'with noisy', len(set(labels))
    unique, counts = np.unique(labels, return_counts=True)

    classes = {}
    for cls in set(labels):
        classes[cls] = []
    
    for i, (ni, nj, fi) in enumerate(feature):
        label = labels[i]
        proba = core_samples_mask[i]
        classes[label].append((ni, nj, proba, fi))

    return np.asarray((unique, counts)).T, classes


def hdbscan_cluster(feature, min_cluster_size=5, min_samples=None, metric='euclidean'):
    
    data = [fi for (ni, nj, fi) in feature]

    if len(data) < min_cluster_size:
        return np.asarray((0,0)), {}

    db = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, metric='euclidean').fit(data)
    labels = db.labels_
    probabilities = db.probabilities_

    # Number of clusters in labels, INCLUDING NOISE -1
    #n_clusters = len(set(labels))
    unique, counts = np.unique(labels, return_counts=True)

    classes = {}
    for cls in set(labels):
        classes[cls] = []
    
    for i, (ni, nj, fi) in enumerate(feature):
        label = labels[i]
        proba = probabilities[i]
        classes[label].append((ni, nj, proba, fi))

    return np.asarray((unique, counts)).T, classes


def agglo_cluster(feature, num_cluster=8, metric='euclidean', linkage='ward'):
    
    data = [fi for (ni, nj, fi) in feature]

    # Affinity = {“euclidean”, “l1”, “l2”, “manhattan”,
    # “cosine”}
    # Linkage = {“ward”, “complete”, “average”}
    Hclustering = AgglomerativeClustering(n_clusters=num_cluster, affinity=metric, linkage=linkage)
    Hclustering.fit(data)
    labels = Hclustering.labels_

    unique, counts = np.unique(labels, return_counts=True)

    classes = {}
    for cls in set(labels):
        classes[cls] = []
    
    for i, (ni, nj, fi) in enumerate(feature):
        label = labels[i]
        classes[label].append((ni, nj, fi))

    return np.asarray((unique, counts)).T, classes
