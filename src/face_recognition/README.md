#Face recognition and clustering pipeline

Downloads videos from YouTube, analyses, stores and clusters results



##Requirements

- directory "facenet" is the Facenet repo including (frozen) network model files, see MtcnnModel.py and FacenetModel.py
    - see https://github.com/davidsandberg/facenet for installation requirements

- directory youtube_dl is part of youtube_dl repo, included here, may need to be updated!

- one of the cython_* versions must be compiled on the used system and the .so-file placed in this directory
    - cython_sparse_arr used (or cython_full_split32 for split computation)

- MySQL credentials must be set in database.py!

Python requirements:
(all only tested for python2)
- OpenCV 2.4.*
- sqlalchemy
- numpy/scipy
- scikit-learn
- pandas
- cPickle
- tensorflow
- HDBSCAN (http://hdbscan.readthedocs.io/en/latest/)
- Cython (cython_* compilation)


##Usage
- the pipeline is split into multiple files

- first video extraction pipeline must be executed, downloading videos and storing features in DB
    - caution: first set source of videos in videoPipeline.py file
    - caution: youtube_dl is used, update youtube_dl directory with current version from github to work correctly
    - start videoPipeline.py

- after video extraction is completed, feature are stored in DB

- clustering the features is next

- depending on number of feature and the system the process is either split up using
on disk memory files or executed at once (could fail due to memory)

- collabDetection.py executes every step in one script
    - when successfull cluster are written in database
    - collaboration graph is then created in collaboration notebook in data_evaluation


IF COMPUTATION TIME IS TOO HIGH, USE SPLIT UP pipeline

- create features array from database
- create indices splitting the matrix up into n parts using collabCreateIndex.py
- compute the parts separate using collabCompIndexBlock.py or collabCompIndexBlockProcess.py

IF MEMORY CONSUMPTION IS TOO HIGH

- try mem-mapped script types, loading the distance matrices as mem-mapped numpy arrays


##Files

tf_board.py:
creates tensorflow data of the models which can be used to browser based tf-board application

test.py:
tests video pipeline, single video test etc.


Directories:

cython_full:
- distance matrix computation using a full distance matrix with size nxn

cython_sparse_arr:
- distance matrix computation using only a sparse distance matrix with size (n*(n-1)/2)

cython_dok_sparse:
- distance matrix computation using a scipy dok sparse matrix object, *not used*