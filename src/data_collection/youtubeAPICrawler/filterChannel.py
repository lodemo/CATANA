# -*- coding: utf-8 -*-

import sys
import codecs
 
from youtubeAPICrawler.database import *

sys.stdout = codecs.getwriter('utf8')(sys.stdout)

db = YTDatabase()

with db._session_scope(True) as session:

    print '#', session.query(Channel).count() # Number of channels in db
    i = 0
    for ch in session.query(Channel): # Check is channel is music related or empty and remove from db

        if any( word in ch.title.lower() for word in [ 'musik', 'vevo', 'music', 'prosieben' 'maxdome', 'weibsbilder', 'the voice', 'games'] ):
            print 'title,-', ch.title, '\n'
            session.delete(ch)

        if any( word in ch.keywords.lower() for word in [ 'musikvideo', 'vevo', 'prosieben', 'sat1', 'sixx', 'maxdome'] ):
            if 'studio71' not in ch.keywords.lower():
                print 'title,-', ch.title
                print 'keyws,-', ch.keywords.lower(), '\n'
                session.delete(ch)
                
        if ch.latestUploadsIDs == '[]':
            session.delete(ch)
            i = i+1

    print 'empty channel:', i
