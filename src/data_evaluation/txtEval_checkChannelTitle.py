# -*- coding: utf-8 -*-

from database import *
import sys
import codecs
import pickle
import pandas as pa
from nltk.corpus import words as nltk_words
import re

# analyse the textual mentions in channel descriptions
# uses blacklist and dictionary for filtering


DIR = '../../data/'

wordset = set(nltk_words.words())

#lacklist = set(['League of Legends', 'the game', 'Mickey Mouse', 'Hasbro', 'Lucas', 'Deni', 'Denis', 'Patrick', 'Richard', 'Jessica', 'dario', 'Powe', 'Japa', 'EMAN', 'Alli']) # use?
blacklist = set(['League of Legends', 'Godzilla', 'the game', 'Mickey Mouse', 'Hasbro', 'Lucas', 'Patrick', 'Richard', 'Jessica', 'Details', 'Reactions']) # use?

def is_english_word(word):
    return word.lower() in wordset

def is_blacklist(word):
    return word in blacklist

def match_title(title, text):
    return re.search(ur'\b{}\b'.format(re.escape(title)), text) is not None

sys.stdout = codecs.getwriter('utf8')(sys.stdout)

try:
    with open(DIR+'channel_to_channel_mentions_with_ID.pkl', 'rb') as input:
        mentions = pickle.load(input)
except:
    mentions = {}
 
db = YTDatabase()

with db._session_scope() as session:

    ch = session.query(Channel).all()

    if len(mentions) == 0:
        for c in ch:
            mentions[c.id] = []

        for c in ch:
            for cd in ch:
                if len(c.title)>3 and not is_blacklist(c.title) and not is_english_word(c.title) and match_title(c.title, cd.description) and c.title != cd.title:
                    #idx = cd.description.find(c.title)
                    print c.title, cd.title, 'MATCH'
                    #print '\t', cd.description[idx-100:idx+len(c.title)], '\n'    
                    mentions[cd.id].append(c.id)
                elif c.id in cd.description and c.title != cd.title:
                    print c.id, cd.title, 'MATCH'
                    mentions[cd.id].append(c.id)

        with open(DIR+'channel_to_channel_mentions_with_ID.pkl', 'wb') as output:
            pickle.dump(mentions, output)

        non_emptys = {}
        for m in mentions:
            if len(mentions[m])>0:
                non_emptys[m]=mentions[m]
        print 'found', len(non_emptys), 'channel with mentions'
    else:
        non_emptys = {}
        for m in mentions:
            if len(mentions[m])>0:
                non_emptys[m]=mentions[m]
        print 'channel with mentions:', len(non_emptys)

        for i in xrange(10):
            c = session.query(Channel).filter(Channel.id == sorted(non_emptys, key=lambda k: len(non_emptys[k]), reverse=True)[i]).first()
            print u'{}'.format(c)
            print len(non_emptys[sorted(non_emptys, key=lambda k: len(non_emptys[k]), reverse=True)[i]])

            for id in non_emptys[sorted(non_emptys, key=lambda k: len(non_emptys[k]), reverse=True)[i]]:
                cd = session.query(Channel).filter(Channel.id == id).first()
                print u'{} ->{}'.format(cd.title, c.title)
                #idx = cd.description.find(c.title)
                print u'\t', cd.description, '\n\n'  