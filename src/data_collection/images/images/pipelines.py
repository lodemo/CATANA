# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: http://doc.scrapy.org/en/latest/topics/item-pipeline.html

import scrapy
from scrapy.pipelines.images import ImagesPipeline
import hashlib

class MyImagesPipeline(ImagesPipeline):

    # name information coming from the spider, in each item
    # add this information to Requests() for individual images downloads
    # through "meta" dict
    def get_media_requests(self, item, info):
        return [scrapy.Request(x, meta={'image_name': item["image_name"]}) for x in item.get('image_urls', [])]


    def file_path(self, request, response=None, info=None):
    
        url = request.url

        #image_guid = hashlib.sha1(to_bytes(url)).hexdigest()  # change to request.url after deprecation
        #return 'full/%s.jpg' % (image_guid)
        
        image_guid = request.url.split('/')[-1]
        return 'full/%s/%s' % (request.meta['image_name'], image_guid)