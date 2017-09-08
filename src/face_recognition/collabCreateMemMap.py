# -*- coding: utf-8 -*-

'''
 Get the features data from DB, de-pickle data and store it into array-on-disk.
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
from sqlalchemy import exists, and_, func


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


from itertools import izip_longest

def find_shape(seq):
    try:
        len_ = len(seq)
    except TypeError:
        return ()
    shapes = [find_shape(subseq) for subseq in seq]
    return (len_,) + tuple(max(sizes) for sizes in izip_longest(*shapes,
                                                                fillvalue=1))

def fill_array(arr, seq):
    if arr.ndim == 1:
        try:
            len_ = len(seq)
        except TypeError:
            len_ = 0
        arr[:len_] = seq
        arr[len_:] = np.nan
    else:
        for subarr, subseq in izip_longest(arr, seq, fillvalue=()):
            fill_array(subarr, subseq)


db = YTDatabase()

start = time.time()

with db._session_scope(False) as session:
    num_features = session.query(VideoFeatures).filter(VideoFeatures.duration > 30.0).count()

print 'ALL features > 30:', num_features


from sqlalchemy.sql import func

videoIDs = []
with db._session_scope(False) as session:
    #vids = session.query(VideoFeatures.videoID).filter(VideoFeatures.duration > 0.0).distinct().all()

    vids = session.query(VideoFeatures.videoID).filter(VideoFeatures.duration > 0.0).filter(and_(Video.id==VideoFeatures.videoID, func.date(Video.crawlTimestamp).between('2016-12-28', '2017-03-28'))).distinct().all()
    print 'first videos:', len(vids)
    for id in vids:
        v = session.query(Video).filter(Video.id==id[0]).first()
        if not v.deleted and session.query(VideoHistory.id).filter(VideoHistory.videoID==v.id).count() > 11:
            videoIDs.append(v.id)


print 'videos', len(videoIDs)


pa.DataFrame(videoIDs).to_csv('videoIDs_filtered.csv')

#videoIDs = pa.read_csv('videoIDs_filtered.csv',index_col=0)['0'].values.tolist()




with db._session_scope(False) as session:
    num_features = session.query(VideoFeatures).filter(and_(VideoFeatures.duration > 30.0, VideoFeatures.videoID.in_(videoIDs))).count()


print 'Video features:', num_features




# This does not work, as the features array is not a perfect array, has different row lengths
#mmap = np.memmap(os.path.join(fileDir, 'features_memmap.npy'), mode='w+', dtype=np.double, shape=(num_features, 100, 1792))

n = 0
with db._session_scope(False) as session:

    # Get features from database, filter duration on query, above 30 sec.
    data =  pa.read_sql(session.query(VideoFeatures.feature).filter(and_(VideoFeatures.duration > 30.0, VideoFeatures.videoID.in_(videoIDs))).statement, db.engine)

    # Feature are stored cpickled
    data['feature'] = data['feature'].apply(cp.loads)

    # We can transform the array into a perfect array but the memory usage will rise aswell
    #fixed_data = np.empty( (len(chunk), 100, 1792) )
    #fill_array(fixed_data, chunk['feature'].values)

    # Due to time consumption of the next computation step, we could only use the first 20 features instead max. 100
    #firstth = np.asarray([f[:20] for f in data['feature'].values])

    # Save features to disk
    np.save('features_3MONTH', np.asarray(data['feature']))
    #mmap[n:n+chunk.shape[0]] = fixed_data
    #n += chunk.shape[0]

processTime = time.time() - start
print 'data extraction took', processTime

