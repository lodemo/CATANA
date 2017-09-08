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
from scrapy import signals
import logging
import urllib
import json
import time
import ushlex

def quoteSplit(value):
    lex = ushlex.shlex(value)
    lex.quotes = '"'
    lex.whitespace_split = True
    lex.commenters = ''
    return list(lex)

from youtubeAPICrawler.items import *

class PopulateYTSpider(scrapy.Spider):
    '''
    Populates the Database with Youtube meta data of the provided channels from channelIDs.txt
    and their related channels see MAX_SEARCH_DEPTH

    API Unit costs: 12 Points per Channel
    '''
    
    name = "populate"
    # use these settings per spider, so different spider different pipelines
    custom_settings = {
        'ITEM_PIPELINES': {
            'youtubeAPICrawler.pipelines.PopulateDatabasePipeline': 900,
        }
    }
    #'youtubeAPICrawler.pipelines.ChannelItemPipeline': 300,
    #'youtubeAPICrawler.pipelines.VideoListItemPipeline': 300,

    allowed_domains = ["www.googleapis.com"]

    MAX_SEARCH_DEPTH = 0 # Crawl the related channel of our provided channel

    YOUTUBE_API_KEY = 'X'
    YOUTUBE_API_CHANNEL_URL = 'https://www.googleapis.com/youtube/v3/channels'
    YOUTUBE_API_SEARCH_URL = 'https://www.googleapis.com/youtube/v3/search'
    YOUTUBE_API_PLAYLISTITEMS_URL = 'https://www.googleapis.com/youtube/v3/playlistItems'  

    '''
    @classmethod
    def from_crawler(cls, crawler, *args, **kwargs):
        spider = super(PopulateYTSpider, cls).from_crawler(crawler, *args, **kwargs)
        crawler.signals.connect(spider.spider_closed, signal=signals.spider_closed)
        return spider


    def spider_closed(self, spider):
        spider.logger.info('Spider closed yoooo: %s', spider.name)
    '''

    def start_requests(self):
        '''
        returns iterable of Requests, either list or generator, which will be begin to crawled
        '''
        urls = []

        with open('bidi_channel_graph_ids.json') as IDs:
            for id in json.load(IDs):
                urls.append(self.generate_channel_request(id))
        
        for url in urls:
            request = scrapy.Request(url=url, callback=self.parseChannel)
            request.meta['search_depth'] = 0
            yield request
    

    def parseChannel(self, response):
        '''
        method to handle the response for each Request made, response holds the page content (for web request)
        '''
        jsonresponse = json.loads(response.body) # change body to text, if encoding issues

        if not jsonresponse['items']:
            logging.info("Empty items in Channel Response.")
            return

        if not 'brandingSettings' in jsonresponse["items"][0]:
            logging.info("Missing item in Channel Response: brandingSettings {}".format(jsonresponse["items"][0]["id"]))
            return
        
        # crawl the related channels until max search depth is reached
        if response.meta['search_depth'] < self.MAX_SEARCH_DEPTH:
            if 'featuredChannelsUrls' in jsonresponse['items'][0]['brandingSettings']['channel']:
                for id in jsonresponse['items'][0]['brandingSettings']['channel']['featuredChannelsUrls']:
                    url = self.generate_channel_request(id)
                    request = scrapy.Request(url, callback=self.parseChannel)
                    request.meta['search_depth'] = response.meta['search_depth'] + 1
                    yield request


        id = jsonresponse["items"][0]["id"]
        title = jsonresponse["items"][0]["brandingSettings"]["channel"]["title"] if 'title' in jsonresponse["items"][0]["brandingSettings"]["channel"] else u'NO TITLE'
        keywords = quoteSplit(jsonresponse["items"][0]["brandingSettings"]["channel"]["keywords"]) if "keywords" in jsonresponse["items"][0]["brandingSettings"]["channel"] else []
        description = jsonresponse["items"][0]["brandingSettings"]["channel"]["description"] if "description" in jsonresponse["items"][0]["brandingSettings"]["channel"] else u''

        if not 'snippet' in jsonresponse["items"][0]:
            dateAdded = ''
        else:
            dateAdded = jsonresponse["items"][0]["snippet"]["publishedAt"] if "publishedAt" in jsonresponse["items"][0]["snippet"] else u''

        featuredChannelsIDs = jsonresponse['items'][0]['brandingSettings']['channel']['featuredChannelsUrls'] if 'featuredChannelsUrls' in jsonresponse['items'][0]['brandingSettings']['channel'] else []

        if not 'contentDetails' in jsonresponse["items"][0]:
            uploadsPlaylistID = ''
        else:
            uploadsPlaylistID = jsonresponse['items'][0]['contentDetails']['relatedPlaylists']['uploads'] if "uploads" in jsonresponse["items"][0]["contentDetails"]["relatedPlaylists"] else u''
        
        unsubscribedTrailer = jsonresponse['items'][0]['brandingSettings']['channel']['unsubscribedTrailer'] if 'unsubscribedTrailer' in jsonresponse['items'][0]['brandingSettings']['channel'] else u''

        if not 'topicDetails' in jsonresponse["items"][0]:
            topicIds = []
        else:
            topicIds = jsonresponse["items"][0]["topicDetails"]["topicIds"] if "topicIds" in jsonresponse["items"][0]["topicDetails"] else []

        # Channels constant infos
        yield ChannelItem(
                id = id,
                title = title,
                keywords = keywords,
                description = description,
                dateAdded = dateAdded,
                featuredChannelsIDs = featuredChannelsIDs,
                uploadsPlaylistID = uploadsPlaylistID,
                unsubscribedTrailer = unsubscribedTrailer,
                topicIds = topicIds,
                crawlTimestamp = time.strftime('%Y-%m-%d %H:%M:%S') 
            )

        if not 'contentDetails' in jsonresponse["items"][0]:
            logging.info("Missing item in Channel Response: contentDetails {}".format(jsonresponse["items"][0]["id"]))
            return
        
        # Get content of the uploads playlist (50 latest videos)
        playlistUrl = self.generate_playlistitems_request(jsonresponse['items'][0]['contentDetails']['relatedPlaylists']['uploads'])
        playlistRequest = scrapy.Request(playlistUrl, callback=self.parsePlaylist)
        playlistRequest.meta['channelID'] = jsonresponse["items"][0]["id"]
        yield playlistRequest


    def parsePlaylist(self, response):
        '''
        method to handle the response for each Request made, response holds the page content (for web request)
        '''
        jsonresponse = json.loads(response.body) # change body to text, if encoding issues

        if not jsonresponse['items']:
            logging.info("Empty items in Playlist Response {}".format(response.meta['channelID']))
            return
            
        videoList = []
        # crawl the playlistitems, videos in the list up to 50 entrys
        for video in jsonresponse['items']:
            videoList.append(video['contentDetails']['videoId'])

        yield VideoListItem(channelID=response.meta['channelID'], videoIDs=videoList)
        

    def generate_channel_request(self, channelID):
        # costs: 11points
        return '{0}?key={1}&id={2}&part=brandingSettings,statistics,contentDetails,snippet,topicDetails&fields=items(id,statistics,topicDetails,snippet(publishedAt),brandingSettings(channel(title,description,keywords,featuredChannelsTitle,featuredChannelsUrls,unsubscribedTrailer)),contentDetails(relatedPlaylists(uploads)))'\
                .format(self.YOUTUBE_API_CHANNEL_URL, self.YOUTUBE_API_KEY, channelID)

    def generate_playlistitems_request(self, playlistID):
        # costs: 3points
        # rather than search for videos in a certain time (1day here), grab the list of uploaded videos on first day, and every added video at the next day in the list is new
        # the playlistitems list is date order it seems, so in the first request should be the 50th newest videos
        return '{0}?key={1}&playlistId={2}&part=contentDetails&maxResults=50'\
                .format(self.YOUTUBE_API_PLAYLISTITEMS_URL, self.YOUTUBE_API_KEY, playlistID)
    