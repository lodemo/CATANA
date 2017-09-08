# -*- coding: utf-8 -*-

'''
 Initials a threadpool for downloading and analysis of YouTube videos.
 Source of videos are specified before e.g. database query or file storage.
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
from ytDownloader import ytDownloader
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

from database import *
from sqlalchemy import exists

import pickle
import csv
import os
import sys
import time


DIR = '../../data/'
NUM_THREADS = 24
CATEGORY_FILTER = [20] # a filter for video selection, category 20=Gaming


def callback(d):
    pass


db = YTDatabase()


#with open(DIR+'txtEval_channel_ID.pkl', 'rb') as input:
#        channelIDs = pickle.load(input)

with db._session_scope(False) as session:
    channelIDs = [r[0] for r in session.query(Channel.id).all()]

videoIDs = []

# Get all video ids from videos in the txtEval mentions and check if deleted or already computed
with db._session_scope(False) as session:
    for cid in channelIDs:
        vs = session.query(Video).filter(Video.channelID==cid).all()
        for v in vs:
            if not session.query(exists().where(VideoFeatures.videoID==v.id)).scalar() and not v.deleted and not int(v.category) in CATEGORY_FILTER:
                videoIDs.append(v.id)

print 'Found {} videos.'.format(len(videoIDs))
 ytd = ytDownloader(callback=callback)

videoURLs = ['https://www.youtube.com/watch?v={}'.format(id) for id in videoIDs]

executor = ThreadPoolExecutor(max_workers=NUM_THREADS)
executor.map(ytd.download, videoURLs, timeout=None)



'''
# not used at the moment, implements a long running job waiting for new crawled videos to be analysed from the database

RETRYS = 6

retry_counter = 0

while True:
    
    if retry_counter > RETRYS:
        sys.exit()

    videoIDs = []

    # get enqueued videos from db with state new
    with db._session_scope(False) as session:
        videos = session.query(VideoFeatureQueue.id).filter(Video.state=='new').all()
        for v in videos:
            videoIDs.append(v.id)

    if len(videoIDs) == 0:
        # wait 12 hours
        time.sleep(43200)
        retry_counter += 1
        continue

    print 'Found {} new enqueued videos.'.format(len(videoIDs))

    ytd = ytDownloader(callback=callback)

    videoURLs = ['https://www.youtube.com/watch?v={}'.format(id) for id in videoIDs]

    executor = ThreadPoolExecutor(max_workers=NUM_THREADS)
    executor.map(ytd.download, videoURLs, timeout=None)
'''