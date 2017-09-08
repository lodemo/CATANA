# -*- coding: utf-8 -*-

from youtubeAPICrawler.database import *

db = YTDatabase()

with db._session_scope(True) as session:
    # Get all channel historys with video counts zero
    zch = session.query(ChannelHistory).filter( (ChannelHistory.videoCount <= 0) | (ChannelHistory.subscriberCount <= 0) |\
        (ChannelHistory.viewCount <= 0) )

    # Delete all channel item in the other table
    for ch in zch:
        print 'found'
        session.query(Channel).filter(Channel.id == ch.channelID).delete(synchronize_session=False)
    
    # Delete the history items
    zch.delete(synchronize_session=False)