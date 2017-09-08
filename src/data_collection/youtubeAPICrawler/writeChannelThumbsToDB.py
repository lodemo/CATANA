# -*- coding: utf-8 -*-
import os.path
import sys
import codecs
import numpy as np
import json

from youtubeAPICrawler.database import *

channelUrl = []
    
with open('channel_thumbnailUrls.json', 'r') as sfile:
    channelUrl = json.loads(sfile.read())


db = YTDatabase()

for (id, url) in channelUrl:
    db.updateChannelThumbnail(id, url)

