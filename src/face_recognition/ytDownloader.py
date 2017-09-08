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

from __future__ import unicode_literals
import youtube_dl

from youtube_dl.utils import ContentTooShortError

import os
import sys

import FacenetModel
import MtcnnModel
import traceback

fileDir = os.path.dirname(os.path.realpath(__file__))
storageDir = '/media/dataStorage/facegraph/'


class MyLogger(object):
    '''
        File logger class writing .log file.
    '''

    ERROR = 0
    WARNING = 1
    DEBUG = 2

    logf = None
    mode = None
    
    def __init__(self, filename, mode=0):
        self.logf = open(filename, 'wb')
        self.mode = mode

    def debug(self, msg):
        if self.mode >= 2:
            self.logf.write(msg)

    def warning(self, msg):
        if self.mode >= 1:
            self.logf.write('{}\n'.format(msg))

    def error(self, msg):
            self.logf.write('{}\n'.format(msg))


class ytDownloader(object):
    '''
    ytDownloader class uses youtube-dl to download YouTube videos through method download.

    callback: function that gets called when video finished
    format: youtube-dl string, format for video downloading, worst, best etc.
    loglevel: loglevel for log file, debug, warning, error

    '''

    callback = None

    output_dir = 'tmp' # TODO use media/datastorage for video storage, even though its just temporary saved
    file_template = '%(id)s.%(ext)s' 

    output_path = os.path.join(output_dir, file_template)

    ydl_opts = {
        'format': '[height<=480]',
        'postprocessors': [{
                'key': 'FacePostProcessor', # A post processor pipeline, applying face recognition on finished videos
                'tfsession': FacenetModel.session,
                'pnet': MtcnnModel.pnet,
                'rnet': MtcnnModel.rnet,
                'onet': MtcnnModel.onet,
                }],
        'logger': None,
        'progress_hooks': [],
        'outtmpl': output_path,
        'restrictfilenames': True,
        'keepvideo': False,
        'logtostderr': True,
    }
    

    def __init__(self, callback, format='mp4[height<=480]/best[height<=480]'):
        self.callback = callback
        self.ydl_opts['progress_hooks'].append(self.youtube_dl_hook)
        self.ydl_opts['format'] = format
        self.ydl_opts['logger'] = MyLogger('ytDownloader.log', mode=MyLogger.WARNING)

    def download(self, url):

        try:
            with youtube_dl.YoutubeDL(self.ydl_opts) as ydl:
                ydl.download([url])
        except Exception as e:
            print 'ytDownloader::download: catched exception'
            print e
            print traceback.print_exc()

    def youtube_dl_hook(self, d):
        if d['status'] == 'finished':
            self.callback(d)
        elif d['status'] == 'error':
            print 'YOUTUBE_DL ERROR: youtube_dl_hook reported error.'
    


