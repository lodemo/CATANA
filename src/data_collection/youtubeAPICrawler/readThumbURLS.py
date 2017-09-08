# -*- coding: utf-8 -*-
import os.path
import sys
import codecs
import numpy as np


from youtubeAPICrawler.database import *

db = YTDatabase()

channelUrl = []

with db._session_scope(True) as session:

    channels = session.query(Channel).all()

    for ch in channels:
        channelUrl.append( (ch.id, ch.thumbnailUrl) )
    
with open('channel_thumbnailUrls.json', 'wb') as sfile:
    sfile.write(json.dumps(channelUrl))
