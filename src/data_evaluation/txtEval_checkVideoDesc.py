# -*- coding: utf-8 -*-

from database import *
import sys
import codecs
import pickle
import pandas as pa
from nltk.corpus import words as nltk_words
import re

# analyse the textual mentions in video titles and descriptions
# uses blacklist and dictionary for filtering

DIR = '../../data/'

wordset = set(nltk_words.words())

blacklist = set(['League of Legends', 'Godzilla', 'Ubisoft', 'Soldiers', 'Reactions', 'Details' 'the game', 'Mickey Mouse', 'Hasbro', 'Lucas', 'Patrick', 'Richard', 'Jessica', 'the game']) # use?

def is_english_word(word):
    return word.lower() in wordset

def is_blacklist(word):
    return word in blacklist

def match_title(title, text):
    return re.search(ur'\b{}\b'.format(re.escape(title)), text) is not None

sys.stdout = codecs.getwriter('utf8')(sys.stdout)

try:
    with open(DIR+'video_to_channel_mentions_with_ID.pkl', 'rb') as input:
        mentions = pickle.load(input)
except:
    mentions = {}

db = YTDatabase()

with db._session_scope() as session:

    ch = session.query(Channel).all()
    vd = session.query(Video).all()

    if len(mentions) == 0:
        for c in ch:
            mentions[c.id] = []

        for c in ch:
            for v in vd:
                if c.id != v.channelID and len(c.title)>3 and not is_blacklist(c.title) and not is_english_word(c.title) and match_title(c.title, v.description):
                    #idx = cd.description.find(c.title)
                    print c.title, v.title, 'MATCH'
                    #print '\t', cd.description[idx-100:idx+len(c.title)], '\n'    
                    mentions[c.id].append(v.id)
                elif c.id != v.channelID and c.id in v.description:
                    print c.id, v.title, 'MATCH'
                    mentions[c.id].append(v.id)

        with open(DIR+'video_to_channel_mentions_with_ID.pkl', 'wb') as output:
            pickle.dump(mentions, output)

        non_emptys = {}
        for m in mentions:
            if len(mentions[m])>0:
                non_emptys[m]=mentions[m]
        print 'found', len(non_emptys),  'videos with mentions'

    else:
        non_emptys = {}
        for m in mentions:
            if len(mentions[m])>0:
                non_emptys[m]=mentions[m]
        print 'channel with mentions:', len(non_emptys)

        for i in xrange(20):
            c = session.query(Channel).filter(Channel.id == sorted(non_emptys, key=lambda k: len(non_emptys[k]), reverse=True)[i]).first()
            print c
            print len(non_emptys[sorted(non_emptys, key=lambda k: len(non_emptys[k]), reverse=True)[i]])

            for id in non_emptys[sorted(non_emptys, key=lambda k: len(non_emptys[k]), reverse=True)[i]]:
                vd = session.query(Video).filter(Video.id == id).first()
		cd = session.query(Channel).filter(Channel.id == vd.channelID).first()
                print u'{}:{} ->{}'.format(cd.title, vd.title, c.title)
                #idx = cd.description.find(c.title)
                #print u'\t', cd.description[idx-100:idx+50], '\n'   

        #for c in ch:
        #    if is_english_word(c.title):
        #        print c.title, is_english_word(c.title)
