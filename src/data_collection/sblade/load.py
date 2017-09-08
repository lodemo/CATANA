# -*- coding: utf-8 -*-

import json
import pprint

with open('network_channel_id.json') as channel_file:    
    channel = json.loads(channel_file.read(), encoding="utf-8")

print len(channel)

pprint.pprint(channel)