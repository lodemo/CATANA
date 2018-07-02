# -*- coding: utf-8 -*-

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
    
    # SQLite based code commented
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
