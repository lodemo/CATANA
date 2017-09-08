# -*- coding: utf-8 -*-

import matplotlib as mpl

mpl.use('Agg')

import matplotlib.pyplot as plt
import pandas as pa 
import json
import os
import pickle
from sqlalchemy import exists
from matplotlib_venn import venn2, venn2_circles
import networkx as nx

# read pkl from channel mentions
# statistics of number of mentions per channel
# compare with featured channel list

from database import *

db = YTDatabase()
DIR = '../../data/'

with db._session_scope(False) as session:

    #num_featured = session.query(FeaturedChannel).count()
    if os.path.isfile(DIR+'networkx_graph_ytDatabase.adjlist'):
        G=nx.read_adjlist(DIR+'networkx_graph_ytDatabase.adjlist', create_using=nx.DiGraph())
        num_featured = G.number_of_edges()

    with open(DIR+'channel_to_channel_mentions_with_ID.pkl', 'rb') as input:
        mentions = pickle.load(input)

    with open(DIR+'video_to_channel_mentions_with_ID.pkl', 'rb') as input:
        videomentions = pickle.load(input)

        for cid in videomentions:
            for vid in videomentions[cid]:
                mid = session.query(Video.channelID).filter(Video.id==vid).first()[0]
                mentions[mid].append(cid)

    non_emptys = {}
    for m in mentions:
        if len(mentions[m])>0:
            non_emptys[m]=mentions[m]
            
    print 'non_emptys:', len(non_emptys)

    channelIDs = set()
    for m in non_emptys:
        chs = non_emptys[m]
        channelIDs.add(m)
        for c in chs:
            channelIDs.add(c)

    print 'channel with mentions:', len(channelIDs)

    with open(DIR+'txtEval_channel_ID.pkl', 'wb') as output:
        pickle.dump(list(channelIDs), output)

    ch = session.query(Channel.id).all()

    channel_intersect = set()
    for c in ch:
        if c[0] in channelIDs:
            channel_intersect.add(c[0])

    print 'all channel:', len(ch)
    print 'channel intersect:', len(channel_intersect)

    videos = []
    for cid in channelIDs:
        vs = session.query(Video).filter(Video.channelID==cid).all()
        for v in vs:
            if not v.deleted:  #and not int(v.category) == 20:
                videos.append(v.id)

    print 'non-deleted videos from channel with mentions:', len(videos)


    num_mentioned = 0
    num_featured_fromMentions = 0
    num_feature_mentions_intersect = 0
    
    for m in non_emptys:
        chms = set(non_emptys[m])
        num_mentioned += len(chms)
        chf = session.query(FeaturedChannel).filter(FeaturedChannel.channelID==m).all()

        chfs = set()
        for cf in chf:
            chfs.add(cf.featuredChannelID)

        num_featured_fromMentions += len(chfs)

        #print len(chms), len(chfs), len(chms.intersection(chfs))
        num_feature_mentions_intersect += len(chms.intersection(chfs))


    plt.figure(0)
    plt.title('Channel links (complete)')
    venn2(subsets = (num_featured, num_mentioned, num_feature_mentions_intersect), set_labels = ('Featured', 'Mentioned', 'Intersect'))
    plt.savefig(DIR+'plots/venn_mention_links.png')

    plt.figure(1)
    plt.title('Mentioned channel')
    venn2(subsets = (len(list(ch)),0, len(list(channel_intersect))), set_labels = ('All', 'Mentioned'))
    plt.savefig(DIR+'plots/venn_mentioned_channel.png')
