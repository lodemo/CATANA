# -*- coding: utf-8 -*-

# detects collaborations of actors from features in db

# read features from db
# method 1
# generate pairs of features (all pairs of not-same video features)
# measure distance of feature lists from pair, like ytf-eval, mean, max etc.
# use threshold (same as ytf-eval?) to decide feature are same person
# if same add edge between feature ids/videoids
# todo how to detect if collab or own channel

# method 2
# cluster features (all feature from every list?)
# add edge between all videos/feature ids in same cluster
# which clustering? dbscan, hdbscan?

# method x -> use clustering with average feature etc?


from __future__ import unicode_literals

import os
import time
import numpy as np
import pandas as pa
import cPickle as cp
import json

import math

import itertools
import string

import networkx as nx

fileDir = os.path.dirname(os.path.realpath(__file__))


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



G_list = {}

G0 = nx.read_gml("sample_cluster_collabs.gml")

G1 = nx.read_gml("hdb_collab_cluster_sample.gml")
G_list['normal'] = G1

G3 = nx.read_gml("hdb_collab_cluster_sample_first.gml")
G_list['first'] = G3

G4 = nx.read_gml("hdb_collab_cluster_sample_mean_mean.gml")
G_list['mean'] = G4

G5 = nx.read_gml("hdb_collab_cluster_sample_median.gml")
G_list['median'] = G5

G6 = nx.read_gml("hdb_collab_cluster_sample_firstth.gml")
G_list['firstth'] = G6

G7 = nx.read_gml("hdb_collab_cluster_sample_firstth20.gml")
G_list['firstth20'] = G7


print 'ground truth sample cluster:'
print 'nodes:', len(G0.nodes())
print 'edges:', len(G0.edges())
print '\n\n'


for node in G1.nodes():
    for key, value in G_list.iteritems():
        if not value.has_node(node):
            value.add_node(node)


for key, value in G_list.iteritems():
    print key, 'sample cluster'
    print 'nodes:', len(value.nodes())
    print 'edges:', len(value.edges())

    print '\nDIFF normal test:'
    print 'missing edges:', len(nx.difference(G1, value).edges())
    print 'added edges:', len(nx.difference(value, G1).edges())
    print '\n\n'



