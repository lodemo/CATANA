# CATANA data evaluation

Data and results of the data evaluation conducted with the CATANA framework. See the paper for a detailed description.


collab_detections_graph.gml:
filtered_collab_detections_graph.gml:
Collaboration graphs found in the data. 
Filtered means no gaming videos are contained.
NetworkX DiGraph based.


network_channel_counts.csv:
Network name and number of channel associated in our data.


yifan_filtered.json:
Gephi graph file, used as input for our interactive visualization.


df_collabs.txt:
df_filtered_collabs.txt:
Actual collaborations found in the data. Described in a "from" and "to" relationship. Channel (content creator) "from" thereby occured in a video of the channel "to".
More information concerning popularity, category and network of both channels is described.

Filtered is the same information but filtering out difficult cases of gaming videos.


df_network_collabs_pairs.txt:
Columns from_network, to_network, nof_collabs
Describing network collaborations in the data.


df_most_collabs_top_pairs.txt:
Channel pairs with most collaborations.


df_most_collabs_top.txt:
Channel with most collaborations.
