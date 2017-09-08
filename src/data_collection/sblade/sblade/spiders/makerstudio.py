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

class MakerStudioSpider(scrapy.Spider):
    name = "makerstudio"
    allowed_domains = ["socialblade.com"]

    file_path = '../../../data/network_channel_id_broadtv.json'

    network_url = 'https://socialblade.com/youtube/network/maker'
    top_url = 'http://socialblade.com/youtube/network/maker/topusers'

    custom_settings = {
            'DOWNLOAD_DELAY': 1.8,
        }

    MAX_ITERATION = 10
    i = 0

    channelIDs = set()

    def start_requests(self):
        '''
        returns iterable of Requests, either list or generator, which will be begin to crawled
        '''

        with open(self.file_path) as IDs:
            for id in json.load(IDs):
                if isinstance(id, list):
                    for i in id:
                        self.channelIDs.add(i)
                else:
                    self.channelIDs.add(id)
        
        self.log('%s old ids read' % len(self.channelIDs))
        
        yield scrapy.Request(self.top_url, callback=self.parseTop, dont_filter=True)
        yield scrapy.Request(self.network_url, callback=self.parse, dont_filter=True)


    def parse(self, response):
        p = re.compile('^/youtube/s/\?q=(.+)$')

        links = response.css('a').xpath('@href').extract()
       
        for ref in links:
            m = p.match(ref)
            if m:
                self.channelIDs.add(m.groups()[0])
        
        if self.i <= self.MAX_ITERATION:
            self.i = self.i + 1
            yield scrapy.Request(response.url, callback=self.parse, dont_filter=True)
        else:
            with open(self.file_path, 'wb') as f:
                f.write(json.dumps(list(self.channelIDs)))
            self.log('Saved {} IDs to file {}'.format(len(self.channelIDs), filename))



    def parseTop(self, response):
        p = re.compile('^/youtube/channel/(.+)$')

        links = response.css('a').xpath('@href').extract()
        
        for ref in links:
            m = p.match(ref)
            if m:
                print 'top:', m.groups()[0]
                self.channelIDs.add(m.groups()[0])