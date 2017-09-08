# -*- coding: utf-8 -*-

'''

Creates the data.js file for using in the javascript visualization.

A exported graph (from gephi) in form of json file must be present in directory, see filename.

'''


from __future__ import unicode_literals

from database import *
from sqlalchemy import exists

import pickle
import csv
import os

import networkx as nx


def get_spaced_colors(n):
    max_value = 16581375 #255**3
    interval = int(max_value / n)
    colors = [hex(I)[2:].zfill(6) for I in range(0, max_value, interval)]

    return ["#{0:02x}{1:02x}{2:02x}".format(int(i[:2], 16), int(i[2:4], 16), int(i[4:], 16)) for i in colors]


DIR = '../../data/'


db = YTDatabase()


# read json file from gephi
# add thumbnail, network information
with open(DIR+'data_evaluation/filtered_yifan_collab_graph.json', 'r') as sfile:
    gdict = json.loads(sfile.read())

gnodes = gdict['nodes']
gedges = gdict['edges']


nodes = []
edges = []
groups = {}
networks = set()

with db._session_scope(True) as session:

    for node in gnodes:
        print node['label']
        ch = {}
        ch['id'] = node['id']
        ch['x'] = node['x']
        ch['y'] = node['y']
        ch['value'] = node['size']

        channel = session.query(Channel).filter(Channel.id==node['label']).first()

        ch['label'] = channel.title
        ch['group'] = channel.network
        networks.add(channel.network)
        if channel.thumbnailUrl:
            ch['shape'] = 'circularImage'
            ch['image'] = 'img/thumbnails/{}.jpg'.format(node['label'])
        nodes.append(ch)

for ed in gedges:
    s = {}
    s['from'] = ed['source']
    s['to'] = ed['target']
    edges.append(s)


colors = get_spaced_colors(len(networks))

for i, n in enumerate(networks):
    nd = {}
    nd['color'] = colors[i]
    groups[n] = nd

with open('filtered_yifan_data.js', 'wb') as dfile:
    dfile.write('var nodes = {}\n'.format(json.dumps(nodes)))
    dfile.write('var edges = {}\n'.format(json.dumps(edges)))
    dfile.write('var groups = {}\n'.format(json.dumps(groups)))

# read G, get channel information from DB
# create data.js file
