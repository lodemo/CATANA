# Channel data

network_channel_id_{broadtv, maker, studio71}.json:
Contain channel-ids of YouTube channel which are associated by the respective YouTube network BroadBandTV, Maker Studios, Studio71.
Associations are extracted from SocialBlade website and may be outdated.

network_channel_id_sampled_of_all3_4.5k.json:
A set of 4.5k channels combined from a sample of each network-set. A random selection of 1.5k channel of each network-set were taken.

networkx_graph_bidi.adjlist:
We constructed a network of channel by creating edges if a channel is contained in another channel's featured-list. This network is the featured-list induced graph, containing only bidrectional edges, thus describing a mutually knowledge between channels.