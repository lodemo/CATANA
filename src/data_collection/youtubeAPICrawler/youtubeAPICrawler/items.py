# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# http://doc.scrapy.org/en/latest/topics/items.html


import scrapy


class ChannelItem(scrapy.Item):
    # This item will be crawled only once, static channel meta infos
    id = scrapy.Field()
    title = scrapy.Field()
    keywords = scrapy.Field()
    description = scrapy.Field()
    dateAdded = scrapy.Field()
    featuredChannelsIDs = scrapy.Field() # will be assumed static for this period
    uploadsPlaylistID = scrapy.Field()
    unsubscribedTrailer = scrapy.Field()
    topicIds = scrapy.Field()
    crawlTimestamp = scrapy.Field()

class ChannelStatisticsItem(scrapy.Item):
    # This item will be crawled multiple times over time, dynamic content
    id = scrapy.Field()
    viewCount = scrapy.Field()
    subscriberCount = scrapy.Field()
    commentCount = scrapy.Field()
    videoCount = scrapy.Field()
    crawlTimestamp = scrapy.Field()


class VideoItem(scrapy.Item):
    # This item will be crawled only once, containing static video meta infos
    id = scrapy.Field()
    channelID = scrapy.Field()
    title = scrapy.Field()
    description = scrapy.Field()
    category = scrapy.Field()
    dateAdded = scrapy.Field()
    tags = scrapy.Field()
    topicIds = scrapy.Field()
    attribution = scrapy.Field() # license, must be crawled from webpage, maybe while downloading(youtube-dl etc.)
    duration = scrapy.Field()
    crawlTimestamp = scrapy.Field()

class VideoStatisticsItem(scrapy.Item):
    # This item will be crawled multiple times over time, dynamic content
    id = scrapy.Field()
    viewCount = scrapy.Field()
    commentCount = scrapy.Field()
    likeCount = scrapy.Field()
    dislikeCount = scrapy.Field()
    crawlTimestamp = scrapy.Field()

class VideoListItem(scrapy.Item):
    channelID = scrapy.Field()
    videoIDs = scrapy.Field()

