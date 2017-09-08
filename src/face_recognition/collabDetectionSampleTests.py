# -*- coding: utf-8 -*-

'''
    Analyses data for a small test sample, creates collab graph
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

THRESHOLD = 1.00


def create_collab_graph(ftcollabs):

    cluster = {}

    label = ftcollabs['label'].unique()

    for l in label:
        ftl = ftcollabs[ftcollabs.label == l]
        groups = ftl.groupby(['channel'])
        vcounts = groups.videoID.nunique()
        cluster[l] = [(cid, nof) for cid, nof in vcounts.sort_values(ascending=False).iteritems()]


    # do graph creation somewhere else, so cluster labels are not lists but single ids generated based on cluster lists?

    G = nx.DiGraph() # undirected graph


    for l, cls in cluster.iteritems():
        mainc = cls[0][0]
        #print mainc
        if G.has_node(mainc):
            if 'cluster' in G.node[mainc]:
                G.node[mainc]['cluster'].append(str(l))
            else:
                G.node[mainc]['cluster'] = [str(l)]
                
                with db._session_scope(False) as session:
                    G.node[mainc]['network'] = session.query(Channel.network).filter(Channel.id == mainc).first()[0]

        else:
            with db._session_scope(False) as session:
                network = session.query(Channel.network).filter(Channel.id == mainc).first()[0]
            G.add_node(mainc, cluster=[str(l)], network=network) # todo make clusters a list, extend list if already there, so multiple cluster could have "main channel"

        for (c, n) in cls[1:]:
            G.add_edge(mainc, c, weight=int(n), cluster=str(l))


    print G.nodes()
    print G.edges()

    #nx.write_gexf(G, "collab_detections_graph.gexf")
    #nx.write_gml(G, "collab_detections_graph.gml")
    
    # save features with labels as pickle ?!
    return G



def hdbscan_tests(features, ftype='mean', min_cluster_size=2):
    
    if ftype=='min':
        D = facedist.min_dist(features)
    elif ftype=='max':
        D = facedist.max_dist(features)
    elif ftype=='meanmin':
        D = facedist.meanmin_dist(features)
    else:
        D = facedist.mean_dist(features)

    print D.shape


    nrow = len(features)
    dense_distances = np.zeros( (nrow, nrow), dtype=np.double)

    for ii in range(nrow):
        for jj in range(ii+1, nrow):
            nn = ii+jj*(jj-1)/2
            rd = D[nn]

            dense_distances[ii, jj] = rd
            dense_distances[jj, ii] = rd

    del D

    db = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric='precomputed').fit(dense_distances)
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

    print D.shape


    nrow = len(features)
    dense_distances = np.zeros( (nrow, nrow), dtype=np.double)

    for ii in range(nrow):
        for jj in range(ii+1, nrow):
            nn = ii+jj*(jj-1)/2
            rd = D[nn]

            dense_distances[ii, jj] = rd
            dense_distances[jj, ii] = rd

    del D

    db = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed').fit(dense_distances)
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


    fto = ft.iloc[:,[0,1,3,4,5,6,7]]

    ftost = fto.sort_values(['channel', 'videoID'])

    return ftost


db = YTDatabase()

start = time.time()


#ftos.to_csv(r'hdb_collab_cluster.txt', sep=str('\t'), encoding='utf-8')
ftoss = pa.read_csv(r'hdb_collab_cluster_small.txt', sep=str('\t'), encoding='utf-8')

channel = ftoss['channel'].unique()

print channel

db = YTDatabase()

start = time.time()

with db._session_scope(False) as session:
    #features = session.query(VideoFeatures).all()
    ft = pa.read_sql(session.query(VideoFeatures).filter((VideoFeatures.videoID==Video.id) & (Video.channelID.in_(channel))).statement, db.engine)

ft = ft[ft.duration > 0.0]
#ft = ft[ft.duration > 28.0]
ft['feature'] = ft['feature'].apply(cp.loads)

features = ft['feature'].values

processTime = time.time() - start
print 'data extraction took', processTime, 'for', features.shape


#np.savetxt('facedist_mean.txt', facedist.mean_dist(features))

#test = facedist.mean_dist(features)
#print describe(test)

#print np.min(test)
#print np.max(test)
#print np.mean(test)

start = time.time()

#uqc, hdb_max_labels, hdb_max_proba = hdbscan_tests(features, ftype='max', min_cluster_size=2)
#uqc, hdb_min_labels, hdb_min_proba, hdb_min_pers = hdbscan_tests(features, ftype='min', min_cluster_size=2)
uqc, hdb_mean_labels, hdb_mean_proba, hdb_mean_pers = hdbscan_tests(features, ftype='mean', min_cluster_size=2)
#uqc, hdb_meanmin_labels, hdb_meanmin_proba = hdbscan_tests(features, ftype='meanmin', min_cluster_size=2)
#print 'hdbscan precomp took', time.time() - start, 'found', uqc

print 'hdbscan normal features took', time.time() - start, 'found', uqc


firstth_features = np.asarray([f[:10] for f in features])

print firstth_features.shape
print firstth_features[0].shape

start = time.time()

uqc, hdb_mean_firstth_labels, hdb_mean_firstth_proba, hdb_mean_firstth_pers = hdbscan_tests(firstth_features, ftype='mean', min_cluster_size=2)
#db = hdbscan.HDBSCAN(min_cluster_size=2, metric='euclidean').fit(firstth_features)
#hdb_mean_firstth_labels = db.labels_
#hdb_mean_firstth_proba = db.probabilities_
#hdb_mean_firstth_pers = db.cluster_persistence_

print 'hdbscan firstth features took', time.time() - start, 'found', uqc


firstth2_features = np.asarray([f[:20] for f in features])

print firstth2_features.shape
print firstth2_features[0].shape

start = time.time()

uqc, hdb_mean_firstth2_labels, hdb_mean_firstth2_proba, hdb_mean_firstth2_pers = hdbscan_tests(firstth2_features, ftype='mean', min_cluster_size=2)
#db = hdbscan.HDBSCAN(min_cluster_size=2, metric='euclidean').fit(firstth_features)
#hdb_mean_firstth_labels = db.labels_
#hdb_mean_firstth_proba = db.probabilities_
#hdb_mean_firstth_pers = db.cluster_persistence_

print 'hdbscan firstth20 features took', time.time() - start, 'found', uqc



first_features = [f[0] for f in features]
print first_features[0].shape

start = time.time()

hdb = hdbscan.HDBSCAN(min_cluster_size=2, metric='euclidean').fit(first_features)
hdb_mean_first_labels = hdb.labels_
hdb_mean_first_proba = hdb.probabilities_
hdb_mean_first_pers = hdb.cluster_persistence_

print 'hdbscan first features took', time.time() - start, 'found', np.asarray(np.unique(hdb_mean_first_labels, return_counts=True)).T


mean_features = [np.mean(f, axis=0) for f in features]
print mean_features[0].shape

start = time.time()

hdb = hdbscan.HDBSCAN(min_cluster_size=2, metric='euclidean').fit(mean_features)
hdb_mean_mean_labels = hdb.labels_
hdb_mean_mean_proba = hdb.probabilities_
hdb_mean_mean_pers = hdb.cluster_persistence_

print 'hdbscan mean features took', time.time() - start, 'found', np.asarray(np.unique(hdb_mean_mean_labels, return_counts=True)).T


med_features = [np.median(f, axis=0) for f in features]
print med_features[0].shape

start = time.time()

hdb = hdbscan.HDBSCAN(min_cluster_size=2, metric='euclidean').fit(med_features)
hdb_mean_med_labels = hdb.labels_
hdb_mean_med_proba = hdb.probabilities_
hdb_mean_med_pers = hdb.cluster_persistence_

print 'hdbscan median features took', time.time() - start, 'found', np.asarray(np.unique(hdb_mean_med_labels, return_counts=True)).T


#ftos = extend_df(ft.copy(), hdb_mean_firstth_labels, hdb_mean_firstth_proba, hdb_mean_firstth_pers)
#ftos = ftos[ftos.pers > 0.05]

# write cluster-feature id pairs into database
#with db._session_scope(True) as session:
#    for row in ftos.itertuples():
#        session.add(VideoFaceCluster(featureID=row.id, cluster=row.label ))

#G2 = create_collab_graph(ftos0)
#nx.write_gml(G2, "hdb_collab_cluster_sample_firstth.gml")


# do tests here

# dbscan with all features,
# dbscan only with first feature from list


#uqc, db_max_labels, _ = dbscan_tests(features, ftype='max', eps=0.7, min_samples=2)
#uqc, db_min_labels, _ = dbscan_tests(features, ftype='min', eps=0.7, min_samples=2)
#uqc, db_mean_labels, _ = dbscan_tests(features, ftype='mean', eps=1.0, min_samples=2)
#uqc, db_meanmin_labels, _ = dbscan_tests(features, ftype='meanmin', eps=0.7, min_samples=2)


#uqc, agg_mean_labels, _ = agglo_tests(features, ftype='mean', num_cluster=int(math.sqrt(len(features))), metric='precomputed')


ftos1 = extend_df(ft.copy(), hdb_mean_labels, hdb_mean_proba, hdb_mean_pers)
ftos1 = ftos1[ftos1.pers > 0.05]
ftos1.to_csv(r'hdb_collab_cluster_sample.txt', sep=str('\t'), encoding='utf-8')

G1 = create_collab_graph(ftos1)
nx.write_gml(G1, "hdb_collab_cluster_sample.gml")



ftosth = extend_df(ft.copy(), hdb_mean_firstth_labels, hdb_mean_firstth_proba, hdb_mean_firstth_pers)
ftosth = ftosth[ftosth.pers > 0.05]
ftosth.to_csv(r'hdb_collab_cluster_sample_firstth.txt', sep=str('\t'), encoding='utf-8')

G2 = create_collab_graph(ftosth)
nx.write_gml(G2, "hdb_collab_cluster_sample_firstth.gml")


ftosth2 = extend_df(ft.copy(), hdb_mean_firstth2_labels, hdb_mean_firstth2_proba, hdb_mean_firstth2_pers)
ftosth2 = ftosth2[ftosth2.pers > 0.05]
ftosth2.to_csv(r'hdb_collab_cluster_sample_firstth20.txt', sep=str('\t'), encoding='utf-8')

G22 = create_collab_graph(ftosth2)
nx.write_gml(G22, "hdb_collab_cluster_sample_firstth20.gml")



ftos2 = extend_df(ft.copy(), hdb_mean_first_labels, hdb_mean_first_proba, hdb_mean_first_pers)
ftos2 = ftos2[ftos2.pers > 0.05]
ftos2.to_csv(r'hdb_collab_cluster_sample_first.txt', sep=str('\t'), encoding='utf-8')

G3 = create_collab_graph(ftos2)
nx.write_gml(G3, "hdb_collab_cluster_sample_first.gml")


ftos3 = extend_df(ft.copy(), hdb_mean_mean_labels, hdb_mean_mean_proba, hdb_mean_mean_pers)
ftos3 = ftos3[ftos3.pers > 0.05]
ftos3.to_csv(r'hdb_collab_cluster_sample_mean_mean.txt', sep=str('\t'), encoding='utf-8')

G4 = create_collab_graph(ftos3)
nx.write_gml(G4, "hdb_collab_cluster_sample_mean_mean.gml")


ftos4 = extend_df(ft.copy(), hdb_mean_med_labels, hdb_mean_med_proba, hdb_mean_med_pers)
ftos4 = ftos4[ftos4.pers > 0.05]
ftos4.to_csv(r'hdb_collab_cluster_sample_median.txt', sep=str('\t'), encoding='utf-8')

G5 = create_collab_graph(ftos4)
nx.write_gml(G5, "hdb_collab_cluster_sample_median.gml")
