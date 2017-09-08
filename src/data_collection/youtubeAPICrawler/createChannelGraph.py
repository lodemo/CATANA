# -*- coding: utf-8 -*-

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

import os.path
import sys
import codecs
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from youtubeAPICrawler.database import *

class ThinDiGraph(nx.DiGraph):
    all_edge_dict = {'weight': 1}
    def single_edge_dict(self):
        return self.all_edge_dict
    edge_attr_dict_factory = single_edge_dict


if os.path.isfile('yt.adjlist'):
    G=nx.read_adjlist('yt.adjlist', create_using=ThinDiGraph())
else:
    db = YTDatabase() # db should be configured to the 16k ytdatabase
    G=ThinDiGraph()

    with db._session_scope(True) as session:

        print '#', session.query(Channel).count() # Number of channels in db

        channels = session.query(Channel).all()

        for channel in channels:
            G.add_node(channel.id)

            for link in channel.featured:
                G.add_edge(link.channelID, link.featuredChannelID)

    nx.write_adjlist(G,"yt.adjlist")

print G.number_of_nodes()
print G.size()

pr = nx.pagerank_scipy(G)

import operator

sorted_pr = sorted(pr.items(), key=operator.itemgetter(1), reverse=True)

sorted_deg = sorted( [(n, G.degree(n)) for n in G.nodes()], key=operator.itemgetter(1), reverse=True)

sorted_degpr = sorted( [(n, G.degree(n)*pr[n]) for n in G.nodes()], key=operator.itemgetter(1), reverse=True)

for ch in sorted_pr[:5]:
    print ch[0], 'pr:', ch[1], 'deg:', G.degree(ch[0]), 'out:', len(G.out_edges(ch[0])), 'in:', len(G.in_edges(ch[0]))

print '...'
for ch in sorted_pr[len(sorted_pr)-5:len(sorted_pr)]:
    print ch[0], 'pr:', pr[ch[0]], 'deg:', G.degree(ch[0]), 'out:', len(G.out_edges(ch[0])), 'in:', len(G.in_edges(ch[0]))

print '\n'

for ch in sorted_deg[:5]:
    print ch[0], 'deg:', ch[1], 'out:', len(G.out_edges(ch[0])), 'in:', len(G.in_edges(ch[0]))

print '...'
for ch in sorted_deg[len(sorted_deg)-5:len(sorted_deg)]:
    print ch[0], 'deg:', ch[1], 'out:', len(G.out_edges(ch[0])), 'in:', len(G.in_edges(ch[0]))

print '\n'


for ch in sorted_degpr[:5]:
    print ch[0], 'deg*pr:', ch[1], 'out:', len(G.out_edges(ch[0])), 'in:', len(G.in_edges(ch[0]))

print '...'
for ch in sorted_degpr[len(sorted_degpr)-5:len(sorted_degpr)]:
    print ch[0], 'deg*pr:', ch[1], 'out:', len(G.out_edges(ch[0])), 'in:', len(G.in_edges(ch[0]))

print '\n'

#with open('page_rank_sampled_channel_id.json', 'wb') as channelid:
#    channelid.write(json.dumps([id[0] for id in sorted_pr[:800]]))

largest = max(nx.strongly_connected_components(G), key=len)
print 'largest strong connected subgraph:', len(largest), '\n'


# trying to get all bidirectional connected nodes only
print '############ BI-DIR ###########\n'

bidirectnodes = []
BG = ThinDiGraph()

for nodeu in G.nodes_iter():
    for nodep in G.neighbors(nodeu):
        if G.has_edge(nodeu, nodep) and G.has_edge(nodep, nodeu):
            BG.add_node(nodeu)
            BG.add_node(nodep)
            BG.add_edge(nodeu, nodep)
            BG.add_edge(nodep, nodeu)
             
print BG.number_of_nodes()
print BG.size()

UG = BG.to_undirected(True) # is the same as BG only undirected
print UG.number_of_nodes()
print UG.size()

with open('bidi_channel_graph_ids.json', 'wb') as sfile:
    sfile.write(json.dumps(UG.nodes()))

nx.write_adjlist(BG,"yt_bidi.adjlist")

prbi = nx.pagerank_scipy(BG)

import operator
sorted_prbi = sorted(prbi.items(), key=operator.itemgetter(1), reverse=True)

sorted_deg = sorted( [(n, BG.degree(n)) for n in BG.nodes()],key=operator.itemgetter(1), reverse=True)

print sorted_prbi[:5]

print '\n'

print sorted_prbi[len(sorted_prbi)-5:len(sorted_prbi)]


for ch in sorted_prbi[:5]:
    print ch[0], 'pr:', ch[1], 'deg:', BG.degree(ch[0]), 'out:', len(BG.out_edges(ch[0])), 'in:', len(BG.in_edges(ch[0]))


for ch in sorted_prbi[len(sorted_prbi)-5:len(sorted_prbi)]:
    print ch[0], 'pr:', prbi[ch[0]], 'deg:', BG.degree(ch[0]), 'out:', len(BG.out_edges(ch[0])), 'in:', len(BG.in_edges(ch[0]))


strongsubgraphs = [Gc for Gc in sorted(nx.strongly_connected_component_subgraphs(BG),key=len, reverse=True)]

print 'biggest strongly connected subgraphs:'
for cs in strongsubgraphs[:5]:
    print 'nodes:', cs.number_of_nodes(),'edges:', cs.size()

#app = Viewer(SBGnn)
#app.mainloop()


#pos=nx.spring_layout(SBG)
#nx.draw(SBG,pos, edge_cmap=plt.cm.Reds)
#plt.show()
