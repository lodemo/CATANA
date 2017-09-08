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
import re
import json
import logging

from youtubeAPICrawler.database import *

class sbladeNetwork(scrapy.Spider):
    name = "networktag"
    allowed_domains = ["socialblade.com"]

    search_url = 'https://socialblade.com/youtube/s/?q='

    custom_settings = {
            'DOWNLOAD_DELAY': 0.5,
            'REDIRECT_ENABLED' : True,
            'REDIRECT_MAX_TIMES': 5,
        }

    db = YTDatabase()

    def start_requests(self):
        '''
        returns iterable of Requests, either list or generator, which will be begin to crawled
        '''
        channelIDs = []

        with self.db._session_scope(True) as session:
            print 'channel#', session.query(Channel).count() # Number of channels in db
            
            for ch in session.query(Channel): # Check is channel is music related or empty and remove from db
                if ch.network == None:
                    channelIDs.append(ch.id)

        logging.info("channel without network:{}".format(len(channelIDs)))

        for id in channelIDs:
            request = scrapy.Request(self.search_url+id, callback=self.parse)
            request.meta['channelID'] = id
            yield request


    def parse(self, response):
        name = response.xpath('//a[@id="youtube-user-page-network"]/text()').extract_first()
        logging.info("found:"+name)

        if name == 'TEST':
            name = 'None'

        with self.db._session_scope(True) as session:
            self.db.updateChannelNetwork(response.meta['channelID'], name)