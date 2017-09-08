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

import scrapy
import logging
import urllib
import json
import time
from datetime import datetime

from youtubeAPICrawler.items import *
from youtubeAPICrawler.database import *


class UpdateYTSpider(scrapy.Spider):
    # get populated ids from DB, and crawl videos and channel statistics here, daily for example

    name = "update"
    # use these settings per spider, so different spider different pipelines
    custom_settings = {
        'ITEM_PIPELINES': {
            'youtubeAPICrawler.pipelines.UpdateDatabasePipeline': 900,
        }
    }

    allowed_domains = ["www.googleapis.com"]

    YOUTUBE_API_KEY = 'X'
    YOUTUBE_API_CHANNEL_URL = 'https://www.googleapis.com/youtube/v3/channels'
    YOUTUBE_API_VIDEO_URL = 'https://www.googleapis.com/youtube/v3/videos'
    YOUTUBE_API_PLAYLISTITEMS_URL = 'https://www.googleapis.com/youtube/v3/playlistItems'  

    def start_requests(self):
        '''
        returns iterable of Requests, either list or generator, which will be begin to crawled
        '''
        channel_playlist_IDs = []
        videoIDs = []

        self.db = YTDatabase()
        with self.db._session_scope(False) as session:
            channel_playlist_IDs = session.query(Channel.id, Channel.uploadsPlaylistID).all()
            videoIDs = session.query(Video.id, Video.deleted).all()

        # get channel ids, create urls, update stats
        for c, p in channel_playlist_IDs:
            request = scrapy.Request(url=self.generate_channelstats_request(c), callback=self.parseChannel)
            yield request

        # get playlists and check for new videos
        for c, p in channel_playlist_IDs:
            request = scrapy.Request(url=self.generate_playlistitems_request(p), callback=self.parsePlaylist)
            request.meta['channelID'] = c
            yield request

        # get already found videos in database and update stats, comma for tuple separating
        for id, deleted in videoIDs:
            if not deleted:
                request = scrapy.Request(url=self.generate_videostats_request(id), callback=self.parseVideo)
                request.meta['videoID'] = id
                yield request


    def parseChannel(self, response):
        '''
        method to handle the response for each Request made, response holds the page content (for web request)
        '''
        jsonresponse = json.loads(response.body) # change body to text, if encoding issues

        if not jsonresponse['items']:
            logging.info("Empty item response for channel ID.")
            return

        # Channels dynamic infos, about ~3-4 credits per channel
        channelStatistics =  ChannelStatisticsItem(
                id = jsonresponse["items"][0]["id"],
                viewCount = jsonresponse["items"][0]["statistics"]["viewCount"],
                subscriberCount = jsonresponse["items"][0]["statistics"]["subscriberCount"],
                commentCount = jsonresponse["items"][0]["statistics"]["commentCount"],
                videoCount = jsonresponse['items'][0]['statistics']["videoCount"],
                crawlTimestamp = time.strftime('%Y-%m-%d %H:%M:%S') 
            )

        self.db.addChannelHistoryEntry(channelStatistics)            


    def parsePlaylist(self, response):
        '''
        method to handle the response for each Request made, response holds the page content (for web request)
        '''
        jsonresponse = json.loads(response.body) # change body to text, if encoding issues

        if not jsonresponse['items']:
            logging.info("Empty items in Playlist Response.")
            return
                    
        # check existing and new videos and filter the new 
        # request crawl for new videos 
        # update new list in db

        #videoPublishedAt

        newVideosIDATE = {}
        newVideos = []
        oldVideos = set()
        with self.db._session_scope(False) as session:
            oldVideos = set(json.loads(session.query(Channel.latestUploadsIDs).filter(Channel.id == response.meta['channelID']).first()[0]))

        # TODO
        # compare the number of items in the playlist with the lastest
        # if its less or equal, exit (no new videos are entered or some even got deleted)
        # save video count with playlist items? latest videocount in history table could be updated already and match everytime

        date_format = '%Y-%m-%dT%H:%M:%S.%fZ'
        crawlTimestamp = datetime.now()

        # crawl the playlistitems, videos in the list up to 50 entrys
        for video in jsonresponse['items']:
            newVideosIDATE[video['contentDetails']['videoId']]=video['contentDetails']['videoPublishedAt']
            newVideos.append(video['contentDetails']['videoId'] )

        diff = set(newVideos).difference(oldVideos)
        if diff:
            logging.info("New Videos found for {}:\n{}".format(response.meta['channelID'], diff))
        else:
            #logging.info("No new Videos found.")
            return

        for id in diff:
            if id in newVideosIDATE:
                publishedAt = datetime.strptime(newVideosIDATE[id], date_format)
                delta = crawlTimestamp - publishedAt
                if delta.days < 3:
                    request = scrapy.Request(url=self.generate_newvideo_request(id), callback=self.parseNewVideo)
                    request.meta['videoID'] = id
                    yield request
                else:
                    logging.info("New Video DATE DIFF FOUND: {} days".format(delta.days))

        self.db.updateLatestUploads(VideoListItem(channelID=response.meta['channelID'], videoIDs=newVideos))


    def parseNewVideo(self, response):
        '''
        method to handle the response for each Request made, response holds the page content (for web request)
        '''
        jsonresponse = json.loads(response.body) # change body to text, if encoding issues

        if not jsonresponse["items"]:
            logging.info("Empty video item response for new video:"+response.meta['videoID'])
            return

        if not 'snippet' in jsonresponse["items"][0] and not 'contentDetails' in jsonresponse["items"][0] and\
        not 'statistics' in jsonresponse["items"][0]:
            logging.info("Missing parts in video response for new video.")
            return

        if not 'topicDetails' in jsonresponse["items"][0]:
            topicIds = []
        else:
            topicIds = jsonresponse["items"][0]["topicDetails"]["topicIds"] if "topicIds" in jsonresponse["items"][0]["topicDetails"] else []

        # TODO check if diff method is really sufficient for new video filter
        # TODO check if parts present, set defaults else
        yield  VideoItem(
                id = jsonresponse["items"][0]["id"],
                channelID = jsonresponse["items"][0]["snippet"]["channelId"],
                title = jsonresponse["items"][0]["snippet"]["title"],
                description = jsonresponse["items"][0]["snippet"]["description"],
                category = jsonresponse['items'][0]['snippet']["categoryId"],
                duration = jsonresponse['items'][0]['contentDetails']["duration"],
                dateAdded = jsonresponse['items'][0]['snippet']["publishedAt"],
                tags = jsonresponse["items"][0]["snippet"]["tags"] if "tags" in jsonresponse["items"][0]["snippet"] else [],
                topicIds = topicIds,
                crawlTimestamp = time.strftime('%Y-%m-%d %H:%M:%S') 
            )

        yield  VideoStatisticsItem(
                id = jsonresponse["items"][0]["id"],
                viewCount = jsonresponse["items"][0]["statistics"]["viewCount"],
                commentCount = jsonresponse["items"][0]["statistics"]["commentCount"] if "commentCount" in jsonresponse["items"][0]["statistics"] else 0,
                likeCount = jsonresponse["items"][0]["statistics"]["likeCount"] if "likeCount" in jsonresponse["items"][0]["statistics"] else 0,
                dislikeCount = jsonresponse['items'][0]['statistics']["dislikeCount"] if "dislikeCount" in jsonresponse["items"][0]["statistics"] else 0,
                crawlTimestamp = time.strftime('%Y-%m-%d %H:%M:%S') 
            )


    def parseVideo(self, response):
        '''
        method to handle the response for each Request made, response holds the page content (for web request)
        '''
        jsonresponse = json.loads(response.body) # change body to text, if encoding issues

        if not jsonresponse['items']:
            logging.info("Empty video item response for existing video:"+response.meta['videoID']+'...setting as deleted')
            self.db.setVideoDeleted(response.meta['videoID'])
            return

        if not 'statistics' in jsonresponse["items"][0]:
            logging.info("Missing parts in Video response.")
            return

        # TODO check if diff method is really sufficient for new video filter
        # TODO check if parts present, set defaults else

        yield  VideoStatisticsItem(
                id = jsonresponse["items"][0]["id"],
                viewCount = jsonresponse["items"][0]["statistics"]["viewCount"],
                commentCount = jsonresponse["items"][0]["statistics"]["commentCount"] if "commentCount" in jsonresponse["items"][0]["statistics"] else 0,
                likeCount = jsonresponse["items"][0]["statistics"]["likeCount"] if "likeCount" in jsonresponse["items"][0]["statistics"] else 0,
                dislikeCount = jsonresponse['items'][0]['statistics']["dislikeCount"] if "dislikeCount" in jsonresponse["items"][0]["statistics"] else 0,
                crawlTimestamp = time.strftime('%Y-%m-%d %H:%M:%S') 
            )



    def generate_channelstats_request(self, channelID):
        # costs: ~3points
        return '{0}?key={1}&id={2}&part=statistics&fields=items(id,statistics)'\
                .format(self.YOUTUBE_API_CHANNEL_URL, self.YOUTUBE_API_KEY, channelID)

    def generate_playlistitems_request(self, playlistID):
        # costs: 3points
        # rather than search for videos in a certain time (1day here), grab the list of uploaded videos on first day, and every added video at the next day in the list is new
        # the playlistitems list is date order it seems, so in the first request should be the 50th newest videos
        return '{0}?key={1}&playlistId={2}&part=contentDetails&maxResults=50'\
                .format(self.YOUTUBE_API_PLAYLISTITEMS_URL, self.YOUTUBE_API_KEY, playlistID)

    def generate_videostats_request(self, videoID):
        # costs: ~3 points
        return '{0}?key={1}&id={2}&part=statistics&fields=items(id,statistics)'\
                .format(self.YOUTUBE_API_VIDEO_URL, self.YOUTUBE_API_KEY, videoID)

    def generate_newvideo_request(self, videoID):
        # costs: ~9 points
        return '{0}?key={1}&id={2}&part=contentDetails,statistics,snippet,topicDetails&fields=items(id,statistics,topicDetails,contentDetails,snippet(publishedAt,channelId,title,tags,description,categoryId))'\
                .format(self.YOUTUBE_API_VIDEO_URL, self.YOUTUBE_API_KEY, videoID)
