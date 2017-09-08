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
import ushlex

def quoteSplit(value):
    lex = ushlex.shlex(value)
    lex.quotes = '"'
    lex.whitespace_split = True
    lex.commenters = ''
    return list(lex)


from youtubeAPICrawler.items import *
from youtubeAPICrawler.database import *


class UpdateYTSpider(scrapy.Spider):
    # get populated ids from DB, and crawl videos and channel statistics here, daily for example

    name = "updatepopulation"
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
            videoIDs = session.query(Video.id).all()

        # get channel ids, create urls, update stats
        for c, p in channel_playlist_IDs:
            request = scrapy.Request(url=self.generate_channel_thumbs_request(c), callback=self.parseChannel)
            yield request

        # get already found videos in database and update stats, comma for tuple separating
        #for id, in videoIDs:
        #    request = scrapy.Request(url=self.generate_newvideo_request(id), callback=self.parseVideo)
        #    yield request
    

    def parseChannel(self, response):
        '''
        method to handle the response for each Request made, response holds the page content (for web request)
        '''
        jsonresponse = json.loads(response.body) # change body to text, if encoding issues

        if not jsonresponse['items']:
            logging.info("Empty items in Channel Response.")
            return

        #if not 'brandingSettings' in jsonresponse["items"][0]:
        #    logging.info("Missing item in Channel Response: brandingSettings {}".format(jsonresponse["items"][0]["id"]))
        #    return

        id = jsonresponse["items"][0]["id"]
        #title = jsonresponse["items"][0]["brandingSettings"]["channel"]["title"] if 'title' in jsonresponse["items"][0]["brandingSettings"]["channel"] else u'NO TITLE'
        #keywords = quoteSplit(jsonresponse["items"][0]["brandingSettings"]["channel"]["keywords"]) if "keywords" in jsonresponse["items"][0]["brandingSettings"]["channel"] else []
        #description = jsonresponse["items"][0]["brandingSettings"]["channel"]["description"] if "description" in jsonresponse["items"][0]["brandingSettings"]["channel"] else u''

        if not 'snippet' in jsonresponse["items"][0]:
            #dateAdded = ''
            print 'NO SNIPPET IN RESPONSE', id
        else:
            
            if not 'thumbnails' in jsonresponse["items"][0]["snippet"]:
                #dateAdded = ''
                print 'NO THUMBNAILS IN RESPONSE', id
            else:
                #dateAdded = jsonresponse["items"][0]["snippet"]["publishedAt"] if "publishedAt" in jsonresponse["items"][0]["snippet"] else u''
                thumbnailUrl = jsonresponse["items"][0]["snippet"]["thumbnails"]["default"]["url"] if "default" in jsonresponse["items"][0]["snippet"]["thumbnails"] else jsonresponse["items"][0]["snippet"]["thumbnails"]["medium"]["url"]

        #featuredChannelsIDs = jsonresponse['items'][0]['brandingSettings']['channel']['featuredChannelsUrls'] if 'featuredChannelsUrls' in jsonresponse['items'][0]['brandingSettings']['channel'] else []

        #if not 'contentDetails' in jsonresponse["items"][0]:
        #    uploadsPlaylistID = ''
        #else:
        #    uploadsPlaylistID = jsonresponse['items'][0]['contentDetails']['relatedPlaylists']['uploads'] if "uploads" in jsonresponse["items"][0]["contentDetails"]["relatedPlaylists"] else u''
        
        #unsubscribedTrailer = jsonresponse['items'][0]['brandingSettings']['channel']['unsubscribedTrailer'] if 'unsubscribedTrailer' in jsonresponse['items'][0]['brandingSettings']['channel'] else u''

        #if not 'topicDetails' in jsonresponse["items"][0]:
        #    topicIds = []
        #else:
        #    topicIds = jsonresponse["items"][0]["topicDetails"]["topicIds"] if "topicIds" in jsonresponse["items"][0]["topicDetails"] else []

        # Channels constant infos
        #citem =  ChannelItem(
        #        id = id,
        #        title = title,
        #        keywords = keywords,
        #        description = description,
        #        dateAdded = dateAdded,
        #        featuredChannelsIDs = featuredChannelsIDs,
        #        uploadsPlaylistID = uploadsPlaylistID,
        #        unsubscribedTrailer = unsubscribedTrailer,
        #        topicIds = topicIds,
        #        crawlTimestamp = time.strftime('%Y-%m-%d %H:%M:%S') 
        #    )

        #self.db.updateChannelEntry(citem)            

        self.db.updateChannelThumbnail(id, thumbnailUrl)

    def parseVideo(self, response):
        '''
        method to handle the response for each Request made, response holds the page content (for web request)
        '''
        jsonresponse = json.loads(response.body) # change body to text, if encoding issues

        if not jsonresponse["items"]:
            logging.info("Empty video item response for new video.")
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
        vitem = VideoItem(
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

        self.db.updateVideoEntry(vitem)            



    def generate_channel_request(self, channelID):
        # costs: 9points
        return '{0}?key={1}&id={2}&part=brandingSettings,statistics,contentDetails,snippet,topicDetails&fields=items(id,statistics,topicDetails,snippet(publishedAt),brandingSettings(channel(title,description,keywords,featuredChannelsTitle,featuredChannelsUrls,unsubscribedTrailer)),contentDetails(relatedPlaylists(uploads)))'\
                .format(self.YOUTUBE_API_CHANNEL_URL, self.YOUTUBE_API_KEY, channelID)

    def generate_channel_thumbs_request(self, channelID):
            # costs: 9points
        return '{0}?key={1}&id={2}&part=snippet&fields=items(id,snippet(thumbnails))'\
                .format(self.YOUTUBE_API_CHANNEL_URL, self.YOUTUBE_API_KEY, channelID)   
               
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
        # costs: ~7 points
        return '{0}?key={1}&id={2}&part=contentDetails,statistics,snippet,topicDetails&fields=items(id,statistics,topicDetails,contentDetails,snippet(publishedAt,channelId,title,tags,description,categoryId))'\
                .format(self.YOUTUBE_API_VIDEO_URL, self.YOUTUBE_API_KEY, videoID)
