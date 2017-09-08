#!/usr/bin/env python2
# -*- coding: utf-8 -*-

# MIT License
# 
# Copyright (c) 2017 Moritz Lode
# Copyright 2015-2016 Carnegie Mellon University
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

'''
From YTF Paper:
1. All pairs comparisons: Each video is represented by
a set of vectors, each one produced by encoding the video
frames using one of a number of existing face descriptors.
Let X1 be the matrix whose columns are the encoding of
the frames of one video, and let X2 be the corresponding
matrix for the other video. We compute a distance matrix
D where Dij = ||X1(:, i) − X2(:, j)||, 
X1(:, i) denotes the i-th column of matrix X1. 
Four basic similarity measures
are then computed using D: the minimum of D, the average
distance, the median distance, and the maximal distance. In
addition we also compute the ‘meanmin’ similarity in which
for each image (of either set) we match the most similar
image from the other set and consider the average of the
distances between the matched pairs.
'''


import cv2
import math
import numpy as np
import pandas as pd
import pickle

from scipy.interpolate import interp1d
from scipy.spatial.distance import cdist, pdist, squareform
from sklearn.cross_validation import KFold
from sklearn.metrics import accuracy_score

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('bmh')

import os
import sys

import argparse

from scipy import arange
from scipy import misc

import tensorflow as tf

import openface
from facenet.src import facenet

sys.path.append("openface")
from openface.helper import mkdirP

fileDir = os.path.dirname(os.path.realpath(__file__))

openfaceDir = os.path.join(fileDir, 'openface')
openfaceModelDir = os.path.join(openfaceDir, 'models', 'openface')
openfaceModelPath = os.path.join(openfaceModelDir, 'nn4.small2.v1.t7')

facenetDir = os.path.join(fileDir, 'facenet')
facenetModelDir = os.path.join(facenetDir, 'models', '20170117-215115')
#facenetModelDir = os.path.join(facenetDir, 'models', '20161116-234200')

splits_path = ''

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'tag', type=str, help='The label/tag to put on the ROC curve.')
    parser.add_argument('workDir', type=str,
                        help='The work directory with labels.csv and reps.csv.')
    parser.add_argument('--lfwMtcnnAligned', type=str,
                        help='lfwMtcnn aligned')
    parser.add_argument('--lfwDlibAligned', type=str,
                        help='lfwMdlib aligned')

    args = parser.parse_args()

    # TODO
    # read splits
    # get pairs, train and test splits 
    # get 50, 100 frames from video pairs and perform face detection, feature extraction
    # calc distances on features, find best threshold for same/not same
    # perform eval on test pairs with best threshold, 100frames(?)

    splits_path = os.path.join(args.workDir, 'splits.txt')

    splits = loadSplits(splits_path)

    cache = os.path.join(args.workDir, 'facenet.ytf.features.pkl')
    facenet_embeddings = cacheToFile(cache)(getFacenetFeatures)(splits, facenetModelDir, args.lfwMtcnnAligned)

    cache = os.path.join(args.workDir, 'openface.ytf.features.pkl')
    openface_embeddings = cacheToFile(cache)(getOpenfaceFeatures)(splits, openfaceModelPath, args.lfwDlibAligned)

    verifyExpFacenet(args.workDir, splits, facenet_embeddings, ['mean', 'min', 'max', 'median', 'meanmin'])
    verifyExpOpenface(args.workDir, splits, openface_embeddings, ['mean', 'min', 'max', 'median', 'meanmin'])
    
    #plotVerifyExp(args.workDir, args.tag)


def cacheToFile(file_name):
    def decorator(original_func):
        global cache
        try:
            cache = pickle.load(open(file_name, 'rb'))
        except:
            cache = None

        def new_func(*param):
            global cache
            if cache is None:
                cache = original_func(*param)
                pickle.dump(cache, open(file_name, 'wb'))
            return cache
        return new_func

    return decorator

def loadSplits(splitsFname):
    print("  + Reading splits.")
    splits = []
    pairs = []
    with open(splitsFname, 'r') as f:
        for line in f.readlines()[1:]:
            pair = [x.strip() for x in line.split(',')]
            print len(pairs), pair
            pairs.append( (pair[2], pair[3], pair[4]) )
            
            if len(pairs) >= 500:
                splits.append(pairs)
                pairs=[]

    assert(len(splits) == 10)
    for p in splits:
        assert(len(p) == 500)
    return np.array(splits)


def getFacenetFeatures(splits, facenetModelDir, lfwAlignedDir):
    print("  + Loading Facenet features.")

    video_features = {}

    # For every video in the pairs, create list for features
    for split in splits:
        for pair in split:
            if not pair[0] in video_features:
                video_features[pair[0]] = []
            if not pair[1] in video_features:
                video_features[pair[1]] = []
    
    with tf.Graph().as_default():
        with tf.Session() as sess:
            # Load the model
            meta_file, ckpt_file = facenet.get_model_filenames(facenetModelDir)
            facenet.load_model(facenetModelDir, meta_file, ckpt_file)

           # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")

            image_size = images_placeholder.get_shape()[1]
            embedding_size = embeddings.get_shape()[1]

            # For every video get the aligned first 100 frames and create features of them
            for video in video_features:
                repCache = {}
                videoDir = os.path.join(lfwAlignedDir, video)
                image_paths = os.listdir(videoDir)
                images = loadFacenetImages(image_paths, videoDir, image_size)

                # Feed single batch of 100 images to network for features
                feed_dict = { images_placeholder:images }
                emb_array = sess.run(embeddings, feed_dict=feed_dict)

                video_features[video] = emb_array
    return video_features

def loadFacenetImages(imgs, imgDir, image_size):
    images = []
    for imgp in imgs[:100]:
        imgPath = os.path.join(imgDir, imgp)
        img = misc.imread(imgPath)

        if img.ndim == 2:
            img = to_rgb(img)

        img = prewhiten(img)
        images.append(crop(img, False, image_size))

    return np.array(images)

def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1/std_adj)
    return y  

def crop(image, random_crop, image_size):
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
  
def flip(image, random_flip):
    if random_flip and np.random.choice([True, False]):
        image = np.fliplr(image)
    return image

def to_rgb(img):
    w, h = img.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
    return ret


def getOpenfaceFeatures(splits, openfaceModel, lfwAlignedDir):
    print("  + Loading Openface features.")
    video_features = {}

    # For every video in the pairs, create list for features
    for split in splits:
        for pair in split:
            if not pair[0] in video_features:
                video_features[pair[0]] = []
            if not pair[1] in video_features:
                video_features[pair[1]] = []

    net = openface.TorchNeuralNet(openfaceModel, 96, cuda=False)

    # For every video get the aligned first 100 frames and create features of them
    for video in video_features:
        repCache = {}
        videoDir = os.path.join(lfwAlignedDir, video)
    image_paths = os.listdir(videoDir)
    images = loadOpenfaceImages(image_paths, videoDir, 96)
    emb_array = []
    for img in images:
        emb = net.forward(img)
        emb_array.append(emb)
    video_features[video] = emb_array

    return video_features


def loadOpenfaceImages(imgs, imgDir, size):
    images = []
    for imgp in imgs[:100]:
        imgPath = os.path.join(imgDir, imgp)
        img = cv2.imread(imgPath)
        img = cv2.resize(img, (size, size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #if img.ndim == 2:
        #    img = to_rgb(img)
        images.append(img)

    return np.array(images)


def getEmbeddings(pair, embeddings):
    
    if len(pair) == 3:
        (x1, x2) = (embeddings[pair[0]], embeddings[pair[1]]) # x1, x2 are actually vectors of features in this case, one for every frame
        actual_same = True if pair[2]==1 else False
        return (x1, x2, actual_same)
    else:
        raise Exception("Unexpected pair length: {}".format(len(pair)))


def writeROC(fname, thresholds, embeddings, pairsTest):
    with open(fname, "w") as f:
        f.write("threshold,tp,tn,fp,fn,tpr,fpr\n")
        tp = tn = fp = fn = 0
        for threshold in thresholds:
            tp = tn = fp = fn = 0
            for pair in pairsTest:
                (x1, x2, actual_same) = getEmbeddings(pair, embeddings)
                diff = x1 - x2
                dist = np.dot(diff.T, diff)
                predict_same = dist < threshold

                if predict_same and actual_same:
                    tp += 1
                elif predict_same and not actual_same:
                    fp += 1
                elif not predict_same and not actual_same:
                    tn += 1
                elif not predict_same and actual_same:
                    fn += 1

            if tp + fn == 0:
                tpr = 0
            else:
                tpr = float(tp) / float(tp + fn)
            if fp + tn == 0:
                fpr = 0
            else:
                fpr = float(fp) / float(fp + tn)
            f.write(",".join([str(x)
                              for x in [threshold, tp, tn, fp, fn, tpr, fpr]]))
            f.write("\n")
            if tpr == 1.0 and fpr == 1.0:
                # No further improvements.
                f.write(",".join([str(x)
                                  for x in [4.0, tp, tn, fp, fn, tpr, fpr]]))
                return


def getDistances(embeddings, split, ftype='mean'):
    list_dist = []
    y_true = []
    for pair in split:
        (x1, x2, actual_same) = getEmbeddings(pair, embeddings) # x1, x2 are actually vectors of features

        #D = np.empty((100, 100))
        #for i, xi in enumerate(x1[:100]):
        #    for j, xj in enumerate(x2[:100]):
        #        diff = xi - xj
        #        dist = np.dot(diff.T, diff)
        #        if float(dist) == 0.0 and i!=j:
        #            print 'PAIR WITH ZERO DISTANCE FOUND:', pair
        #        D[i,j] = dist
        #print x1[:100].shape, x2[:100].shape
        x1 = np.array(x1)
        x2 = np.array(x2)

        if len(x1) == 0 or len(x2) == 0:
            #list_dist.append(float("inf"))
            #y_true.append(actual_same)
            continue

        try:
            D = cdist(x1[:100], x2[:100], metric='euclidean')
        except:
            print x1[:100].shape, x2[:100].shape, pair
            sys.exit()
        #D = squareform(D)

        if len(np.where(D == 0)[0]) > 0:
            print 'PAIR WITH ZERO DISTANCE FOUND:', pair

        if ftype=='min':
            list_dist.append(np.min(D))
        elif ftype=='max':
            list_dist.append(np.max(D))
        elif ftype=='median':
            list_dist.append(np.median(D))
        elif ftype=='meanmin':
            # meanmin:
            # get min of every coloumn/row, mean from these
            mins = D.min(axis=1)
            #print mins.size
            mean = mins.mean()
            list_dist.append(mean)
        else:
            list_dist.append(np.mean(D))

        y_true.append(actual_same)
    return np.asarray(list_dist), np.array(y_true)


def evalThresholdAccuracy(embeddings, splitsTest, threshold, ftype='mean'):
    bad = []
    for split in splitsTest:
        distances, y_true = getDistances(embeddings, split, ftype)
        y_predict = np.zeros(y_true.shape)
        y_predict[np.where(distances < threshold)] = 1

        y_true = np.array(y_true)
        accuracy = accuracy_score(y_true, y_predict)
        bad.append(split[np.where(y_true != y_predict)])
    return accuracy, bad


def findBestThreshold(thresholds, embeddings, splitsTrain, ftype='mean'):
    bestThresh = bestThreshAcc = -1
    for split in splitsTrain:
        distances, y_true = getDistances(embeddings, split, ftype)
        y_true = np.array(y_true)
        for threshold in thresholds:
            y_predlabels = np.zeros(y_true.shape)
            y_predlabels[np.where(distances < threshold)] = 1
            accuracy = accuracy_score(y_true, y_predlabels)
            if accuracy >= bestThreshAcc:
                if threshold == 0.0 and bestThreshAcc != -1:
                    print 'ZERO THRESHOLD FOUND\n', split
                bestThreshAcc = accuracy
                bestThresh = threshold
            else:
                # No further improvements for this split
                break
                #return bestThresh # We dont exit complete here as we iterate through splits too
        return bestThresh


def verifyExpFacenet(workDir, splits, embeddings, ftypes=[]):
    print("  + Computing accuracy with:", ftypes)
    folds = KFold(n=10, n_folds=10, shuffle=False) # this is given in ytf, 10 splits, 5000 pairs
    print 'Folds:', folds
    thresholds = arange(0, 4, 0.01)

    for ftype in ftypes:
        accuracies = []
        with open("{}/facenet_accuracies_{}.txt".format(workDir, ftype), "w") as f:
            f.write('fold, threshold, accuracy\n')
            for idx, (train, test) in enumerate(folds): # train, test are splits indices not pairs here
                #fname = "{}/l2-roc.fold-{}.csv".format(workDir, idx)
                #writeROC(fname, thresholds, embeddings, splits[test])

                bestThresh = findBestThreshold(thresholds, embeddings, splits[train], ftype)

                accuracy, pairs_bad = evalThresholdAccuracy(embeddings, splits[test], bestThresh, ftype)

                accuracies.append(accuracy)
                f.write('{}, {:0.2f}, {:0.2f}\n'.format(
                    idx, bestThresh, accuracy))
            
            avg = np.mean(accuracies)
            std = np.std(accuracies)
            f.write('\navg, {:0.4f} +/- {:0.4f}\n'.format(avg, std))
            print('    + {:0.4f}'.format(avg))


def verifyExpOpenface(workDir, splits, embeddings, ftypes=[]):
    print("  + Computing accuracy with:", ftypes)
    folds = KFold(n=10, n_folds=10, shuffle=False) # this is given in ytf, 10 splits, 5000 pairs
    thresholds = arange(0, 4, 0.01)

    for ftype in ftypes:
        accuracies = []
        with open("{}/openface_accuracies_{}.txt".format(workDir, ftype), "w") as f:
            f.write('fold, threshold, accuracy\n')
            for idx, (train, test) in enumerate(folds): # train test are splits not pairs here
                #fname = "{}/l2-roc.fold-{}.csv".format(workDir, idx)
                #writeROC(fname, thresholds, embeddings, splits[test])

                bestThresh = findBestThreshold(thresholds, embeddings, splits[train], ftype)

                accuracy, pairs_bad = evalThresholdAccuracy(embeddings, splits[test], bestThresh, ftype)

                accuracies.append(accuracy)
                f.write('{}, {:0.2f}, {:0.2f}\n'.format(
                    idx, bestThresh, accuracy))
            avg = np.mean(accuracies)
            std = np.std(accuracies)
            f.write('\navg, {:0.4f} +/- {:0.4f}\n'.format(avg, std))
            print('    + {:0.4f}'.format(avg))


def getAUC(fprs, tprs):
    sortedFprs, sortedTprs = zip(*sorted(zip(*(fprs, tprs))))
    sortedFprs = list(sortedFprs)
    sortedTprs = list(sortedTprs)
    if sortedFprs[-1] != 1.0:
        sortedFprs.append(1.0)
        sortedTprs.append(sortedTprs[-1])
    return np.trapz(sortedTprs, sortedFprs)


def plotOpenFaceROC(workDir, plotFolds=True, color=None):
    fs = []
    for i in range(10):
        rocData = pd.read_csv("{}/l2-roc.fold-{}.csv".format(workDir, i))
        fs.append(interp1d(rocData['fpr'], rocData['tpr']))
        x = np.linspace(0, 1, 1000)
        if plotFolds:
            foldPlot, = plt.plot(x, fs[-1](x), color='grey', alpha=0.5)
        else:
            foldPlot = None

    fprs = []
    tprs = []
    for fpr in np.linspace(0, 1, 1000):
        tpr = 0.0
        for f in fs:
            v = f(fpr)
            if math.isnan(v):
                v = 0.0
            tpr += v
        tpr /= 10.0
        fprs.append(fpr)
        tprs.append(tpr)
    if color:
        meanPlot, = plt.plot(fprs, tprs, color=color)
    else:
        meanPlot, = plt.plot(fprs, tprs)
    AUC = getAUC(fprs, tprs)
    return foldPlot, meanPlot, AUC


def plotVerifyExp(workDir, tag):
    print("Plotting.")

    fig, ax = plt.subplots(1, 1)

    openbrData = pd.read_csv("comparisons/openbr.v1.1.0.DET.csv")
    openbrData['Y'] = 1 - openbrData['Y']
    # brPlot = openbrData.plot(x='X', y='Y', legend=True, ax=ax)
    brPlot, = plt.plot(openbrData['X'], openbrData['Y'])
    brAUC = getAUC(openbrData['X'], openbrData['Y'])

    foldPlot, meanPlot, AUC = plotOpenFaceROC(workDir, color='k')

    humanData = pd.read_table(
        "comparisons/kumar_human_crop.txt", header=None, sep=' ')
    humanPlot, = plt.plot(humanData[1], humanData[0])
    humanAUC = getAUC(humanData[1], humanData[0])

    deepfaceData = pd.read_table(
        "comparisons/deepface_ensemble.txt", header=None, sep=' ')
    dfPlot, = plt.plot(deepfaceData[1], deepfaceData[0], '--',
                       alpha=0.75)
    deepfaceAUC = getAUC(deepfaceData[1], deepfaceData[0])

    # baiduData = pd.read_table(
    #     "comparisons/BaiduIDLFinal.TPFP", header=None, sep=' ')
    # bPlot, = plt.plot(baiduData[1], baiduData[0])
    # baiduAUC = getAUC(baiduData[1], baiduData[0])

    eigData = pd.read_table(
        "comparisons/eigenfaces-original-roc.txt", header=None, sep=' ')
    eigPlot, = plt.plot(eigData[1], eigData[0])
    eigAUC = getAUC(eigData[1], eigData[0])

    ax.legend([humanPlot, dfPlot, brPlot, eigPlot,
               meanPlot, foldPlot],
              ['Human, Cropped [AUC={:.3f}]'.format(humanAUC),
               # 'Baidu [{:.3f}]'.format(baiduAUC),
               'DeepFace Ensemble [{:.3f}]'.format(deepfaceAUC),
               'OpenBR v1.1.0 [{:.3f}]'.format(brAUC),
               'Eigenfaces [{:.3f}]'.format(eigAUC),
               'OpenFace {} [{:.3f}]'.format(tag, AUC),
               'OpenFace {} folds'.format(tag)],
              loc='lower right')

    plt.plot([0, 1], color='k', linestyle=':')

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    # plt.ylim(ymin=0,ymax=1)
    plt.xlim(xmin=0, xmax=1)

    plt.grid(b=True, which='major', color='k', linestyle='-')
    plt.grid(b=True, which='minor', color='k', linestyle='-', alpha=0.2)
    plt.minorticks_on()
    # fig.savefig(os.path.join(workDir, "roc.pdf"))
    fig.savefig(os.path.join(workDir, "roc.png"))

if __name__ == '__main__':
    main()


