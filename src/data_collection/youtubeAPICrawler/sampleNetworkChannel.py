import pandas as pa 
import json
from os import path

data_dir = '../../../data/'

files = ['network_channel_id_studio71.json', 'network_channel_id_maker.json', 'network_channel_id_broadtv.json']

sample_ids = []

SAMPLE = True
SAMPLE_SIZE = 1500

for file in files:
    print path.join(data_dir, file)

    with open(path.join(data_dir, file)) as chfile:
        channel = pa.read_json(chfile, orient = 'records')

    if SAMPLE and channel is not None:
        sample = channel.sample(n=SAMPLE_SIZE)
        
        for chn in sample.values:
            sample_ids.append(chn[0])
    elif not SAMPLE and channel is not None:
        for chn in channel.values:
            sample_ids.append(chn[0])

print len(sample_ids)
with open('sampled_channel_id.json', 'wb') as sfile:
    sfile.write(json.dumps(sample_ids))