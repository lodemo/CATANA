#Collaboration Graph Visualization


Visualization uses the script from Gugel Universum.
http://universum.gugelproductions.de/


##Requirements
- thumbnail URLs must be stored with the channels in the DB
- a json file must be created using the Gephi graph tool containing the needed information

##Usage
- download the thumbnails using getThumbnails.py
- create data.js file using createDatafile.py and the json file exported from Gephi
- execute python -m SimpleHTTPServer inside this directory to host Visualization on localhost