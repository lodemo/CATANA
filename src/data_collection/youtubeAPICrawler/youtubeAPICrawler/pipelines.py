# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: http://doc.scrapy.org/en/latest/topics/item-pipeline.html


import json
from youtubeAPICrawler.items import *



class PopulateDatabasePipeline(object):
    
    def __init__(self):
        # init, create new database or open existing database
        from youtubeAPICrawler import database
        self.db = database.YTDatabase()

    def open_spider(self, spider):
        pass

    def close_spider(self, spider):
        pass

    def process_item(self, item, spider):
        # add new channel and create feature relationships
        if isinstance(item, ChannelItem):
            self.db.addChannelEntry(item)
        elif isinstance(item, VideoListItem):
            self.db.updateLatestUploads(item)

class FilterChannelItemsPipeline(object):
    
    def __init__(self):
        pass

    def open_spider(self, spider):
        pass

    def close_spider(self, spider):
        pass

    def process_item(self, item, spider):
        # if the channel does not meet our terms, drop and not save it in db
        if isinstance(item, ChannelItem):
            if 'vevo' in item['title']:
                raise DropItem("Dropped Channel with vevo in Title: %s" % item['id'])
        return item



class UpdateDatabasePipeline(object):
    
    def __init__(self):
        # init, open existing database
        from youtubeAPICrawler import database
        self.db = database.YTDatabase()

    def open_spider(self, spider):
        pass

    def close_spider(self, spider):
        pass

    def process_item(self, item, spider):
        # create new statistic update entry for channel and videos
        if isinstance(item, ChannelStatisticsItem):
            self.db.addChannelHistoryEntry(item)
        elif isinstance(item, VideoItem):
            self.db.addVideoEntry(item)
            self.db.enqueueVideo(item)
        elif isinstance(item, VideoStatisticsItem):
            self.db.addVideoHistoryEntry(item)

class FilterVideoItemsPipeline(object):
    
    def __init__(self):
        pass

    def open_spider(self, spider):
        pass

    def close_spider(self, spider):
        pass

    def process_item(self, item, spider):
        # if the channel does not meet our terms, drop and not save it in db
        if isinstance(item, VideoItem):
            if item['category'] == 20: # Music
                raise DropItem("Dropped Video with Category Music: %s" % item['id'])
        return item
