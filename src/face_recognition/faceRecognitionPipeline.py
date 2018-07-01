# -*- coding: utf-8 -*-

'''
 Implements the video pipeline and pre-clustering
 Videos are opened, analysed, feature clustered and returned
'''

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

import os
import time
from shutil import copy
import traceback
import sys
import gc
from guppy import hpy

import cv2
import numpy as np
from scipy import misc
import tensorflow as tf
import math

from clustering import hdbscan_cluster, dbscan_cluster
from facenet.src.align import detect_face

fileDir = os.path.dirname(os.path.realpath(__file__))

#from sizes import total_size

class FaceRecognitionPipeline(object):

    session = None

    pnet = None
    rnet = None
    onet = None
    
    minsize = 60 # minimum size of face default, else set with frame height?
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor

    margin = 32
    image_size = 160

    max_feature_per_class = 100
    max_frames = 8000
    min_cluster_size = 10
    actual_min_cluster_size = 15
    min_sample_size = 5


    batch_size = 200

    def __init__(self, fnsession, pnet, rnet, onet):

        self.session = fnsession
        self.pnet = pnet
        self.rnet = rnet
        self.onet = onet

        # Load/open Video file
        # Select shots/frames with metric from Video
        # Detect faces in frames
        # calc feature from frame
        # return features

    #@profile
    def getFeaturesFromVideo(self, videoPath):
        # load video file from path

        try:
            video = cv2.VideoCapture(videoPath) # Open video with opencv

            if video.isOpened():
                frame_width = video.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
                frame_height = video.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)
                frame_num = video.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
                frame_rate = video.get(cv2.cv.CV_CAP_PROP_FPS)

                frame_sec = int(frame_num / frame_rate)
                face_size = int( 1.0/6.0 * frame_height) # for height 360 thats 60

                print frame_width,'x',frame_height
                print frame_num,'@', frame_rate
                print frame_sec,'s'
                print 'face size:', face_size

                self.minsize = face_size

                debug_path = os.path.splitext(videoPath)[0]
                #os.mkdir(debug_path)

                def index_spread(flength, stime):
                    fnum = int(((1.0/6.0)* stime) + 600)
                    if fnum >= self.max_frames:
                        fnum = self.max_frames
                    return np.linspace(0, flength-1, num=fnum, dtype=int)

                features = self._extractFramesToFeatures(video, index_spread)
                #frames = self._extractFrames(video, index_spread)
                #aligned = self._detectFaces(frames, debug_path)
                #features = self._extractFeatures(aligned)
                video.release()
                gc.collect()

                _, cluster_cls = hdbscan_cluster(features, min_cluster_size=self.min_cluster_size, min_samples=self.min_sample_size, metric='euclidean')

                if len(cluster_cls) <= 1: # Only -1 noise cluster found
                    print 'DBSCAN FALLBACK'
                    _, cluster_cls = dbscan_cluster(features, eps=0.7, min_samples=self.min_cluster_size, metric='euclidean')


                # discard -1 label noise cluster
                cluster_cls.pop(-1, None)

                classes = []
                for cls in cluster_cls:
                    feature = cluster_cls[cls]
                    if len(feature) >= self.actual_min_cluster_size:
                        # Duration estimation using frame numbers and framerate
                        timestamps = [ti for (ti, tj, p, f) in feature]
                        dists = [b-a for a, b in self._pairwise(timestamps)]
                        md = np.mean(dists)
                        duration = sum([i for i in dists if i<=md]) + len(timestamps) # If higher skips in frames are present, we try to eliminate these with mean comparison
                        duration = float(duration / frame_rate)

                        # Max feature cut off with sorted proba
                        feature.sort(key=lambda tup: tup[2], reverse=True)
                        feature = feature[:self.max_feature_per_class]
                        feature = [f for (ni, nj, p, f) in feature]
                        feature = np.array(feature, copy=False)
                        classes.append((duration, feature))
            else:
                print 'FaceRecognitionPipeline::getFeaturesFromVideo: Video could not be opened'
                classes = None
                video.release()
                gc.collect()

        except Exception as e:
            print 'FaceRecognitionPipeline::getFeaturesFromVideo: catched exception'
            print e
            print traceback.print_exc()
            classes = None
        
        return classes


    def _pairwise(self, iterable):
        it = iter(iterable)
        a = next(it, None)

        for b in it:
            yield (a, b)
            a = b

    def classesToDir(self, out_dir, name, classes):
        dir_name = os.path.join(out_dir, name)
        print dir_name
        os.mkdir(dir_name)
        if type(classes) is dict:
            classes = [item[1] for item in classes.items()]
        classes = [cls for cls in classes if len(cls) >= self.actual_min_cluster_size]

        for i, cls in enumerate(classes):
            #cls = classes[label]
            class_dir = os.path.join(dir_name, '{}'.format(i))
            #print class_dir
            os.mkdir(class_dir)
            for (ni, nj, p, fi) in cls:
                copy(os.path.join(out_dir, 'test{}_{}.png'.format(ni, nj)), os.path.join(class_dir, 'test{}_{}_{}.png'.format(ni, nj, p)))


    #@profile
    def _extractFramesToFeatures(self, video, metricFn):
        # We combined every step into one function to lower memory consumption by not accumulating all frames for every next step

        # metricFn is function for selecting frame indices
        frame_num = video.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
        frame_rate = video.get(cv2.cv.CV_CAP_PROP_FPS)

        slength = int(frame_num / frame_rate)

        indices = metricFn(frame_num, slength)
        start = time.time()
        print 'Extracting indices', len(indices)

        alignedFrames = []
        features = []
        for i in indices:
            video.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, i) # sets video pointer to frame i
            ret, frame = video.read() # reads frame 
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                bounding_boxes, _ = detect_face.detect_face(frame, self.minsize, self.pnet, self.rnet, self.onet, self.threshold, self.factor)
                nrof_faces = bounding_boxes.shape[0]
                #print 'faces,shape:', nrof_faces, np.array(points).shape # points are (10, n) array for n faces

                # resize to embedding_size according to bounding boxes
                if nrof_faces > 0:
                    dets = bounding_boxes[:,0:4]
                    img_size = np.asarray(frame.shape)[0:2]
                    for j, det in enumerate(dets):
                        det = np.squeeze(det)
                        bb = np.zeros(4, dtype=np.int32)
                        bb[0] = np.maximum(det[0]-self.margin/2, 0)
                        bb[1] = np.maximum(det[1]-self.margin/2, 0)
                        bb[2] = np.minimum(det[2]+self.margin/2, img_size[1])
                        bb[3] = np.minimum(det[3]+self.margin/2, img_size[0])
                        cropped = frame[bb[1]:bb[3],bb[0]:bb[2],:]

                        # Filtering based on blurriness deactivated as it didnt perform quite good
                        #if self._sum_of_magnitude(gray) < self.sharpness_threshold:
                        #    discarded +=1
                        #    continue  # if the cropped image is blurred, discard
                        scaled = misc.imresize(cropped, (self.image_size, self.image_size), interp='bilinear')
                        #scaled = cv2.resize(cropped, (self.image_size, self.image_size))

                        scaled = self._preprocess(scaled) # PREWHITEN ETC
                        #print scaled.shape, scaled.dtype
                        alignedFrames.append((i, j, scaled))

                if len(alignedFrames)>=self.batch_size:
                    with self.session.graph.as_default():
                        with self.session.as_default():
                            # Get input and output tensors
                            images_placeholder = tf.get_default_graph().get_tensor_by_name("resnet/input:0")
                            embeddings = tf.get_default_graph().get_tensor_by_name("resnet/embeddings:0")
                            #phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("resnet/phase_train:0")

                            imgs = np.array([f for (i, j, f) in alignedFrames], copy=False)

                            feed_dict = { images_placeholder: imgs } #phase_train_placeholder: False }
                            emb = self.session.run(embeddings, feed_dict=feed_dict)

                            features.extend([(i, j, e) for (i, j, f), e in zip(alignedFrames, emb)])
                            alignedFrames = []
                    
        
        if len(alignedFrames) > 0:
            with self.session.graph.as_default():
                with self.session.as_default():
                    # Get input and output tensors
                    images_placeholder = tf.get_default_graph().get_tensor_by_name("resnet/input:0")
                    embeddings = tf.get_default_graph().get_tensor_by_name("resnet/embeddings:0")
                    #phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("resnet/phase_train:0")

                    imgs = np.array([f for (i, j, f) in alignedFrames], copy=False)

                    feed_dict = { images_placeholder: imgs } #phase_train_placeholder: False }
                    emb = self.session.run(embeddings, feed_dict=feed_dict)

                    features.extend([(i, j, e) for (i, j, f), e in zip(alignedFrames, emb)])


        del alignedFrames
        gc.collect()

        #print total_size(features)

        processTime = time.time() - start
        print 'feature extraction took', processTime, 's, found', len(features)

        return features


    def _extractFrames(self, video, metricFn):
        # metricFn is function for selecting frame indices
        frame_num = video.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
        frame_rate = video.get(cv2.cv.CV_CAP_PROP_FPS)

        slength = int(frame_num / frame_rate)

        indices = metricFn(frame_num, slength)
        start = time.time()
        print 'Extracting indices', len(indices)

        frames = []
        for i in indices:
            video.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, i) # sets video pointer to frame i
            ret, frame = video.read() # reads frame 
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                #print ret, frame.shape
                frames.append((i, frame))
                
        processTime = time.time() - start
        print 'frame extraction took', processTime, 's ,', len(frames), 'frames'
        print 'frames size: ', sys.getsizeof(frames)
        return frames
    

    def _detectFaces(self, frames, out_dir):
        start = time.time()

        discarded = 0
        alignedFrames = []
        for i, frame in frames:
            bounding_boxes, _ = detect_face.detect_face(frame, self.minsize, self.pnet, self.rnet, self.onet, self.threshold, self.factor)
            nrof_faces = bounding_boxes.shape[0]
            #print 'faces,shape:', nrof_faces, np.array(points).shape # points are (10, n) array for n faces

            # resize to embedding_size according to bounding boxes
            if nrof_faces > 0:
                dets = bounding_boxes[:,0:4]
                img_size = np.asarray(frame.shape)[0:2]
                for j, det in enumerate(dets):
                    det = np.squeeze(det)
                    bb = np.zeros(4, dtype=np.int32)
                    bb[0] = np.maximum(det[0]-self.margin/2, 0)
                    bb[1] = np.maximum(det[1]-self.margin/2, 0)
                    bb[2] = np.minimum(det[2]+self.margin/2, img_size[1])
                    bb[3] = np.minimum(det[3]+self.margin/2, img_size[0])
                    cropped = frame[bb[1]:bb[3],bb[0]:bb[2],:]

                    # Filtering based on blurriness deactivated as it didnt perform quite good
                    #if self._sum_of_magnitude(gray) < self.sharpness_threshold:
                    #    discarded +=1
                    #    continue  # if the cropped image is blurred, discard
                    scaled = misc.imresize(cropped, (self.image_size, self.image_size), interp='bilinear')
                    #scaled = cv2.resize(cropped, (self.image_size, self.image_size))

                    scaled = self._preprocess(scaled) # PREWHITEN ETC
                    alignedFrames.append((i, j, scaled))
                    #misc.imsave(os.path.join(out_dir, 'test{}_{}.png'.format(i, j)), scaled)

        processTime = time.time() - start
        print 'face detection for', len(frames), 'took', processTime, 's, ~', processTime/len(frames),'s; found',len(alignedFrames)

        return alignedFrames


    def _sum_of_magnitude(self, image):
        gx = cv2.Sobel(image, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(image, cv2.CV_32F, 0, 1)
        dm = cv2.magnitude(gx, gy)
        return np.sum(dm)


    def _preprocess(self, image):
        
        image = self._prewhiten(image)
        if image.shape[1] > self.image_size:
            image = self._crop(image, False, self.image_size)
        return image

    def _prewhiten(self, x):
        mean = np.mean(x)
        std = np.std(x)
        std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
        y = np.multiply(np.subtract(x, mean), 1/std_adj)
        return y  

    def _crop(self, image, random_crop, image_size):
        if image.shape[1]>image_size:
            sz1 = image.shape[1]//2
            sz2 = image_size//2
            if random_crop:
                diff = sz1-sz2
                (h, v) = (np.random.randint(-diff, diff+1), np.random.randint(-diff, diff+1))
            else:
                (h, v) = (0,0)
            image = image[(sz1-sz2+v):(sz1+sz2+v),(sz1-sz2+h):(sz1+sz2+h),:]
        return image


    def _extractFeatures(self, alignedFrames):
        start = time.time()

        if len(alignedFrames) <= 0:
            return []

        with self.session.graph.as_default():
            with self.session.as_default():
                # Get input and output tensors
                images_placeholder = tf.get_default_graph().get_tensor_by_name("resnet/input:0")
                embeddings = tf.get_default_graph().get_tensor_by_name("resnet/embeddings:0")
                #phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("resnet/phase_train:0")

                #image_size = images_placeholder.get_shape()[1]
                #embedding_size = embeddings.get_shape()[1]

                frames = [f for (i, j, f) in alignedFrames]

                imgs = np.array(frames, copy=False)
                feed_dict = { images_placeholder: imgs } #phase_train_placeholder: False }
                emb = self.session.run(embeddings, feed_dict=feed_dict)
                #print emb.shape
                features = [(i, j, e) for (i, j, f), e in zip(alignedFrames, emb)]

        processTime = time.time() - start
        print 'feature extraction for', len(alignedFrames), 'took', processTime, 's, ~', processTime/len(alignedFrames)
        return features
    
