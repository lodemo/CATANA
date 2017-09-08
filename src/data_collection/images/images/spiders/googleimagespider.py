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
import re
import urllib
from urlparse import urlparse, parse_qs

from images.items import *

class GoogleImageSpider(scrapy.Spider):
    
    name = "images"

    search_url = 'http://www.google.de/search?q={}&source=lnms&tbm=isch'


    custom_settings = {
        'ITEM_PIPELINES': {
            'images.pipelines.MyImagesPipeline': 100,
        }
    }  

    NUM_IMAGES = 10;

    db = YTDatabase()

    def start_requests(self):
        '''
        returns iterable of Requests, either list or generator, which will be begin to crawled
        '''
        channel = []

        with self.db._session_scope(True) as session:
            print 'channel#', session.query(Channel).count() # Number of channels in db
            
            for ch in session.query(Channel): # filter channel first?
                channel.append( (ch.id, ch.title) )

        # also download youtube thumbnail from every channel!?
        # filter images based on face detection in crawler or separate?
        # resize to big images!
        
        # iterate db channel names here, create requests based on titles
        for id, title in channel:
            url = '{}{}'.format(self.search_url, title)
            request = scrapy.Request(url=url, callback=self.parse)
            request.meta['id'] = id
            yield request
    

    def parse(self, response):
        '''
        method to handle the response for each Request made, response holds the page content (for web request)
        '''
        #print response.body
        # rg_di rg_bx rg_el ivg-i
        urls= []
        item = ImagesItem()
        item['image_urls'] = []

        i = 0
        for image in response.css("div.ivg-i"):
            if i >= self.NUM_IMAGES:
                break
            i = i+1

            ah = image.css("a::attr(href)").extract_first()
            url =  parse_qs(urlparse(ah).query)['imgurl'][0]
            print url
            item['image_urls'].append(url)
            item['image_name'] = response.meta['id']

        return item