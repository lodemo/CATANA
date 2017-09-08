# -*- coding: utf-8 -*-

import sys
import codecs
import os.path

from youtubeAPICrawler.database import *

#sys.stdout = codecs.getwriter('utf8')(sys.stdout)

db = YTDatabase()

data_dir = '../../../data/'

studio71 = 'network_channel_id_studio71.json'
maker = 'network_channel_id_maker.json'
broadtv = 'network_channel_id_broadtv.json'

channel71 = []
channelMaker = []
channelBroad = []

with open(path.join(data_dir, studio71)) as chfile:
    channel71.extend(json.load(chfile))

print '71:', len(channel71)

with open(path.join(data_dir, maker)) as chfile:
    channelMaker.extend(json.load(chfile))

print 'maker:', len(channelMaker)

with open(path.join(data_dir, broadtv)) as chfile:
    channelBroad.extend(json.load(chfile))

print 'broad:', len(channelBroad)

i = 0
with db._session_scope(True) as session:

    print 'channel#', session.query(Channel).count() # Number of channels in db
    
    for ch in session.query(Channel).all():

        if ch.id in channel71:
            db.updateChannelNetwork(ch.id, 'Studio71')
        elif ch.id in channelMaker:
            db.updateChannelNetwork(ch.id, 'Maker_Studios')
        elif ch.id in channelBroad:
            db.updateChannelNetwork(ch.id, 'BroadbandTV')
        else:
            i+=1

    print 'channel with no matching network:', i
