# Detection and Analysis of Content Creator Collaborations in Youtube Videos using Face Recognition

YouTube analysis concerning content creator collaborations in videos, using face recognition.
This projects includes data acquisition from YouTube ,video and face processing and statistics evaluation.

# Directories

## data
contains evaluation results and plots

## src
contains code for the different pipeline steps

### data_collection
crawler code for YouTube, Socialblade and Google Images

### data_evaluation
evaluation code, jupyter-notebook based evaluation, code for plot creation

### face_recognition
video download, face recognition and clustering pipeline, contains Facenet code base aswell as youtube_dl

### visualization
collaboration graph visualization based on Gugel Universum (http://universum.gugelproductions.de/).

### external
misc scripts, face recognition evaluation scripts (heavily based on FaceNet and OpenFace (https://cmusatyalab.github.io/openface/) code)


# Face Recognition Algorithm
For face feature extraction trained models and code from [FaceNet](https://github.com/davidsandberg/facenet) is used.
Face detection and alignment uses MTCNN [Multi-task CNN](https://kpzhang93.github.io/MTCNN_face_detection_alignment/index.html).


## Clustering
For clustering face features, the algorithm [HDBSCAN](https://github.com/scikit-learn-contrib/hdbscan) is used.

## Performance
Face recognition evaluation were conducted on the [YouTube Faces](https://www.cs.tau.ac.il/~wolf/ytfaces/) dataset aswell as Labeled Faces in the Wild [LFW](http://vis-www.cs.umass.edu/lfw/) dataset.

|     | accuracy |
|-----|----------|
| LFW | 0.993 +- 0.004   |
| YTF | 0.998 +- 0.0013  |


# Data
Data for applying face recognition and evaluation can be acquired using the provided crawler in [data_collection] directory.


# Usage

## Installation
todo

## Requirements
see requirements.txt