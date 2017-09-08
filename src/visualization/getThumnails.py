# -*- coding: utf-8 -*-

'''
Downloads all channel thumbnails given a json file with channel-id and thumbnail URL crawled before-hand.

'''


import os
import sys
import codecs
import numpy as np
import json

import urllib


channelUrl = []
    
with open('channel_thumbnailUrls.json', 'r') as sfile:
    channelUrl = json.loads(sfile.read())

for (id, url) in channelUrl:
    if url and not os.path.exists("img/thumbnails/{}.jpg".format(id)):  
        urllib.urlretrieve(url, "img/thumbnails/{}.jpg".format(id))

