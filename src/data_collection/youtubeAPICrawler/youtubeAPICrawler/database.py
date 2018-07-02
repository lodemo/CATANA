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

from sqlalchemy import Column, ForeignKey, Integer, BigInteger, String, Table, UnicodeText, Unicode, Boolean, LargeBinary, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy import create_engine
from sqlalchemy import UniqueConstraint

from sqlalchemy.orm import sessionmaker
from contextlib import contextmanager
import json
from os import path #posixpath

Base = declarative_base()


'''
''
'    Table Definitions
''
'''

class Channel(Base):
    __tablename__ = 'channel'

    id = Column(String(24), primary_key=True) # string may not work, use other then
    title = Column(Unicode(100), nullable=False)
    keywords = Column(UnicodeText, nullable=False) #a space-separated list of strings.
    description = Column(UnicodeText) # max 1000chars
    dateAdded = Column(String(24), nullable=False)
    uploadsPlaylistID = Column(String(24), nullable=False)
    latestUploadsIDs = Column(String(750), nullable=False) # json list string
    unsubscribedTrailer = Column(String(11), nullable=False)
    topicIds = Column(String(131))
    network = Column(String(42)) # for later addition
    crawlTimestamp = Column(String(20), nullable=False)
    thumbnailUrl = Column(String(300))

    history = relationship('ChannelHistory', cascade='delete')
    videos = relationship('Video', cascade='delete')
    featured = relationship('FeaturedChannel', cascade='delete')

    def __repr__(self):
        return u'<Channel (name=|%s|)>' % self.title


class FeaturedChannel(Base):
    __tablename__ = 'channel_featured'

    id = Column(Integer, primary_key=True)
    channelID = Column(String(24), ForeignKey('channel.id'))
    featuredChannelID = Column(String(24))


class ChannelHistory(Base):
    __tablename__ = 'channel_history'

    id = Column(Integer, primary_key=True)
    channelID = Column(String(24), ForeignKey('channel.id'))
    viewCount = Column(BigInteger, nullable=False)
    subscriberCount = Column(Integer, nullable=False)
    commentCount = Column(Integer, nullable=False)
    videoCount = Column(Integer, nullable=False)
    crawlTimestamp = Column(String(20), nullable=False)


class Video(Base):
    __tablename__ = 'video'

    id = Column(String(11), primary_key=True) # string may not work, use other then
    channelID = Column(String(24), ForeignKey('channel.id'))
    title = Column(Unicode(300), nullable=False)
    description = Column(UnicodeText, nullable=False) # max ~5000 characters actually
    category = Column(Integer, nullable=False)
    dateAdded = Column(String(24), nullable=False)
    tags = Column(Unicode(750), nullable=False) # max 500 characters
    topicIds = Column(String(131))
    attribution = Column(String(42)) # for later network attribution
    duration = Column(String(20), nullable=False)
    crawlTimestamp = Column(String(20), nullable=False) # datetime type
    deleted = Column(Boolean)

    history = relationship('VideoHistory')
    feature = relationship('VideoFeatures', cascade='delete')


class VideoHistory(Base):
    __tablename__ = 'video_history'

    id = Column(Integer, primary_key=True)
    videoID = Column(String(11), ForeignKey('video.id'))
    viewCount = Column(Integer, nullable=False)
    commentCount = Column(Integer, nullable=False)
    likeCount = Column(Integer, nullable=False)
    dislikeCount = Column(Integer, nullable=False)
    crawlTimestamp = Column(String(20), nullable=False)


class VideoFeatures(Base):
    __tablename__ = 'video_features'

    id = Column(Integer, primary_key=True)
    videoID = Column(String(11), ForeignKey('video.id'))
    feature = Column(LargeBinary) # correct datatype for numpy/pandas array? test
    duration = Column(Float) # correct datatype for numpy/pandas array? test

    cluster = relationship('VideoFaceCluster', cascade='delete')


class VideoFaceCluster(Base):
    __tablename__ = 'video_face_cluster'

    id = Column(Integer, primary_key=True)
    featureID = Column(Integer, ForeignKey('video_features.id'))
    cluster = Column(Integer)


class VideoFeatureQueue(Base):
    __tablename__ = 'video_feature_queue'

    id = Column(String(11), primary_key=True)
    state = Column(String(9))



'''
''
'    Database API class
''
'''


class YTDatabase(object):

    #DATA_DIR = '/../../../data/'
    #DB_FILE = 'ytDatabase.db'

    DB_NAME = 'X'

    DB_USER = 'X'
    DB_PW = 'X'

    DB_HOST = '127.0.0.1'
    DB_PORT = '3306'

    
    def __init__(self):
        #DB_PATH = path.join(self.DATA_DIR, self.DB_FILE)
        #self.engine = create_engine('sqlite://'+DB_PATH, encoding='utf-8', convert_unicode=True)

        # This engine just used to query for list of databases
        mysql_engine = create_engine('mysql+mysqldb://{0}:{1}@{2}:{3}'.format(self.DB_USER, self.DB_PW, self.DB_HOST, self.DB_PORT), encoding='utf-8', convert_unicode=True)

        # Query for existing databases
        mysql_engine.execute("CREATE DATABASE IF NOT EXISTS {0} ".format(self.DB_NAME))

        # Go ahead and use this engine
        self.engine = create_engine('mysql+mysqldb://{0}:{1}@{2}:{3}/{4}?charset=utf8mb4'.format(self.DB_USER, self.DB_PW, self.DB_HOST, self.DB_PORT, self.DB_NAME), encoding='utf-8', convert_unicode=True)


        Base.metadata.bind = self.engine
        self.DBSession = sessionmaker(bind = self.engine)
        self.createDatabase()
        self.DBSession().execute("SET NAMES utf8mb4 COLLATE 'utf8mb4_unicode_ci'")
        self.DBSession().execute("SET CHARACTER SET utf8mb4")

    def createDatabase(self, drop=False, update=False):
        if drop:
            Base.metadata.drop_all()
        if not self.engine.table_names(): # checks if no tables exists
            Base.metadata.create_all()
        elif update:
            Base.metadata.create_all()


    @contextmanager
    def _session_scope(self, commit=False):
        """Provide a transactional scope around a series of operations."""
        session = self.DBSession()
        try:
            yield session
            if commit:
                session.commit()
        except:
            session.rollback()
            raise
        finally:
            session.close()   

    # TODO handle if already entry with id, sqlalchemy raises exception, integrityerror
    def addChannelEntry(self, channelItem):
        with self._session_scope(True) as session:
            channel = Channel(
                id=channelItem['id'],
                title=channelItem['title'],
                keywords=u'{}'.format(json.dumps(channelItem['keywords'])),
                description=channelItem['description'], # use zlib to compress string?
                dateAdded=channelItem['dateAdded'],
                uploadsPlaylistID=channelItem['uploadsPlaylistID'],
                latestUploadsIDs=json.dumps([]), # will be updated later
                topicIds=json.dumps(channelItem['topicIds'][:10], separators=(',',':')),
                unsubscribedTrailer = channelItem['unsubscribedTrailer'],
                crawlTimestamp=channelItem['crawlTimestamp'])

            for id in channelItem['featuredChannelsIDs']:
                fc = FeaturedChannel(channelID=channelItem['id'], featuredChannelID=id)
                session.add(fc)
                channel.featured.append(fc)

            session.add(channel)

    def updateChannelEntry(self, channelItem):
        with self._session_scope(True) as session:
            session.query(Channel).filter(Channel.id == channelItem['id'])\
            .update({
                    "title": channelItem['title'],
                    "keywords": u'{}'.format(json.dumps(channelItem['keywords'])),
                    "description": channelItem['description'],
                    "topicIds": json.dumps(channelItem['topicIds'][:10], separators=(',',':')),
                    "unsubscribedTrailer": channelItem['unsubscribedTrailer']})

    def updateChannelThumbnail(self, id, thumbnailUrl):
         with self._session_scope(True) as session:
            session.query(Channel).filter(Channel.id == id)\
            .update({
                    "thumbnailUrl": thumbnailUrl
                    })

    def addChannelHistoryEntry(self, channelStatisticsItem):
        with self._session_scope(True) as session:
            channelStats = ChannelHistory(
                channelID=channelStatisticsItem['id'],
                viewCount=channelStatisticsItem['viewCount'],
                subscriberCount=channelStatisticsItem['subscriberCount'],
                commentCount=channelStatisticsItem['commentCount'],
                videoCount=channelStatisticsItem['videoCount'],
                crawlTimestamp=channelStatisticsItem['crawlTimestamp'])
            session.add(channelStats)

    def updateLatestUploads(self, videoListItem):
        with self._session_scope(True) as session:
            session.query(Channel).filter(Channel.id == videoListItem['channelID'])\
            .update({'latestUploadsIDs': json.dumps(videoListItem['videoIDs'], separators=(',',':'))})


    def updateChannelNetwork(self, channelID, networkName):
        with self._session_scope(True) as session:
            session.query(Channel).filter(Channel.id == channelID)\
            .update({'network': networkName})
        

    def addVideoEntry(self, videoItem):
        with self._session_scope(True) as session:
            video = Video(
                id=videoItem['id'],
                channelID=videoItem['channelID'],
                title=videoItem['title'],
                description=videoItem['description'], # use zlib to compress string?
                category=videoItem['category'],
                duration=videoItem['duration'],
                dateAdded=videoItem['dateAdded'],
                tags=u'{:.750}'.format(json.dumps(videoItem['tags'], separators=(',',':'))),
                topicIds=json.dumps(videoItem['topicIds'][:10], separators=(',',':')),
                crawlTimestamp=videoItem['crawlTimestamp'])

            if 'attribution' in videoItem:
                video.attribution = videoItem['attribution']

            session.add(video)

    def updateVideoEntry(self, videoItem):
        with self._session_scope(True) as session:
            session.query(Video).filter(Video.id == videoItem['id'])\
            .update({
                "title": videoItem['title'],
                "description": videoItem['description'], # use zlib to compress string?,
                "tags": u'{:.750}'.format(json.dumps(videoItem['tags'], separators=(',',':'))),
                "topicIds": json.dumps(videoItem['topicIds'][:10], separators=(',',':'))})

    def setVideoDeleted(self, videoID):
        with self._session_scope(True) as session:
            session.query(Video).filter(Video.id == videoID)\
            .update({
                "deleted": True})


    def addVideoHistoryEntry(self, videoStatisticsItem):
        with self._session_scope(True) as session:
            videoStats = VideoHistory(
                videoID=videoStatisticsItem['id'],
                viewCount=videoStatisticsItem['viewCount'],
                commentCount=videoStatisticsItem['commentCount'],
                likeCount=videoStatisticsItem['likeCount'],
                dislikeCount=videoStatisticsItem['dislikeCount'],
                crawlTimestamp=videoStatisticsItem['crawlTimestamp'])
            session.add(videoStats)

    def enqueueVideo(self, videoItem):
        with self._session_scope(True) as session:
            videoQ = VideoFeatureQueue(
                videoID=videoItem['id'],
                state='new')
            session.add(videoQ)

    def updateQueue(self, videoID, state):
        with self._session_scope(True) as session:
            session.query(VideoFeatureQueue).filter(VideoFeatureQueue.id == videoID)\
            .update({
                "state": state})
