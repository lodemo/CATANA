# -*- coding: utf-8 -*-

'''
 Implements a youtube_dl postprocessor class
 Caution: to work with youtube_dl this class must be added to the postprocessor modules __init__ file! 
 see youtube_dl/postprocessor directory
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
from youtube_dl.postprocessor.common import PostProcessor
from youtube_dl.utils import PostProcessingError

import os
import time
import cPickle as cp

import gc

from database import *

from faceRecognitionPipeline import FaceRecognitionPipeline

fileDir = os.path.dirname(os.path.realpath(__file__))

db = YTDatabase()

class FacePostProcessorPP(PostProcessor):

    facePipeline = None


    def __init__(self, downloader, tfsession, pnet, rnet, onet):
        super(FacePostProcessorPP, self).__init__(downloader)

        # initialize face recog pipeline here with tf stuff, call in run
        self.facePipeline = FaceRecognitionPipeline(tfsession, pnet, rnet, onet)

    #@profile
    def run(self, information):
        print 'start POSTPROCESS: ', information['filepath']
        start = time.time()


        vidpath = os.path.join(fileDir, information['filepath'])

        # Get features from given video file, returned as clustered classes
        classes = self.facePipeline.getFeaturesFromVideo(vidpath)

        # classes are None, if error with video file occured, dont write anything into db
        if classes is None:
            #with db._session_scope(True) as session:
                #db.updateQueue(videoID=information['id'], state='error')
            return [information['filepath']], information

        with db._session_scope(True) as session:
            if len(classes) > 0:
                # write feature classes for video id in database
                for cls in classes:
                    session.add(VideoFeatures(videoID=information['id'], duration=cls[0], feature=cp.dumps(cls[1], protocol=2)))
                    #db.updateQueue(videoID=information['id'], state='completed')
            else:
                # Empty classes case, write null entry in database to know its already computed (no faces/people were found in video)
                session.add(VideoFeatures(videoID=information['id'], duration=0, feature=None))
                #db.updateQueue(videoID=information['id'], state='empty')

        processTime = time.time() - start
        print 'end POSTPROCESS: ', information['filepath'], 'took', processTime, '\nresults n_classes', len(classes)

        return [information['filepath']], information # first argument are files to be deleted after post process

