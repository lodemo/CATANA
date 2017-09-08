
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

from threading import Thread
import cv2
import numpy as np
import pandas as pd

from math import ceil

from sklearn.svm import SVC, LinearSVC
from sklearn.cross_validation import ShuffleSplit
from sklearn.metrics import accuracy_score

from scipy import misc

import tensorflow as tf

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('bmh')

import operator
import os
import pickle
import sys
import time

import argparse

import openface
from facenet.src import facenet

sys.path.append("openface")
from openface.helper import mkdirP

fileDir = os.path.dirname(os.path.realpath(__file__))

openfaceDir = os.path.join(fileDir, 'openface')
openfaceModelDir = os.path.join(openfaceDir, 'models', 'openface')
openfaceModelPath = os.path.join(openfaceModelDir, 'nn4.small2.v1.t7')

facenetDir = os.path.join(fileDir, 'facenet')
#facenetModelDir = os.path.join(facenetDir, 'models', '20161116-234200')
facenetModelDir = os.path.join(facenetDir, 'models', '20170117-215115')

nPplVals = [10, 20, 50] #, 200, 400] #, 1000, 1500]
nImgs = 20

cmap = plt.get_cmap("Set1")
#colors = cmap(np.linspace(0, 0.5, 6))
colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
alpha = 0.7

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--lfwDlibAligned', type=str,
                        help='Location of Dlib aligned LFW images')
    parser.add_argument('--lfwMtcnnAligned', type=str,
                    help='Location of MTCNN aligned LFW images')
    parser.add_argument('--largeFont', action='store_true')
    parser.add_argument('workDir', type=str,
                        help='The work directory where intermediate files and results are kept.')
    args = parser.parse_args()
    # print(args)

    if args.largeFont:
        font = {'family': 'normal', 'size': 20}
        mpl.rc('font', **font)

    mkdirP(args.workDir)

    # get ytf feature pickles
    #print("Getting OpenFace YTF features")
    #cache = os.path.join(args.workDir, 'facenet.ytf.features.pkl')
    #openface_ytf = cacheToFile(cache)(getFacenetFeatures)(facenetModelDir, args.lfwMtcnnAligned)

    #print("Getting Facenet YTF features")
    #cache = os.path.join(args.workDir, 'openface.ytf.features.pkl')
    #facenet_ytf = cacheToFile(cache)(getOpenfaceFeatures)(openfaceModelPath, args.lfwDlibAligned)

    ytf_dlib = get_dataset(args.lfwDlibAligned)
    ytf_mtcnn = get_dataset(args.lfwMtcnnAligned)


    print("OpenFace SVM Experiment")
    net = openface.TorchNeuralNet(openfaceModelPath, 96, cuda=False)
    cls = SVC(C=1, kernel='linear')
    cache = os.path.join(args.workDir, 'openface.cpu.svm.pkl')
    openfaceCPUsvmDf = cacheToFile(cache)(openfaceExp)(ytf_dlib, net, cls)

    print("OpenFace LinearSVC Experiment")
    net = openface.TorchNeuralNet(openfaceModelPath, 96, cuda=False)
    cls = LinearSVC(C=1, multi_class='ovr')
    cache = os.path.join(args.workDir, 'openface.cpu.linearsvm.pkl')
    openfaceCPUlinearsvmDf = cacheToFile(cache)(openfaceExp)(ytf_dlib, net, cls)

    print("Facenet SVM Experiment")
    cls = SVC(C=1, kernel='linear')
    cache = os.path.join(args.workDir, 'facenet.svm.pkl')
    facenetsvmDf = cacheToFile(cache)(facenetExp)(ytf_mtcnn, facenetModelDir, cls)

    print("Facenet LinearSVC Experiment")
    cls = LinearSVC(C=1, multi_class='ovr')
    cache = os.path.join(args.workDir, 'facenet.linearsvm.pkl')
    facenetlinearsvmDf = cacheToFile(cache)(facenetExp)(ytf_mtcnn, facenetModelDir, cls)


    plotAccuracy(args.workDir, args.largeFont, openfaceCPUsvmDf, openfaceCPUlinearsvmDf, facenetsvmDf, facenetlinearsvmDf)
    plotTrainingTime(args.workDir, args.largeFont, openfaceCPUsvmDf, openfaceCPUlinearsvmDf, facenetsvmDf, facenetlinearsvmDf)
    plotPredictionTime(args.workDir, args.largeFont, openfaceCPUsvmDf, openfaceCPUlinearsvmDf, facenetsvmDf, facenetlinearsvmDf)

# http://stackoverflow.com/questions/16463582


def get_dataset(paths):
    dataset = []
    classlen = {}
    for path in paths.split(':'):
        path_exp = os.path.expanduser(path)
        classes = os.listdir(path_exp)
        print('Number of classes:', len(classes))
        classes.sort()
        nrof_classes = len(classes)
        for pclass in classes:
            classDir = os.path.join(path_exp, pclass)
            if os.path.isdir(classDir):
                videos = os.listdir(classDir)
                classlen[classDir] = len(videos)
                dataset.append(classDir) #(classDir, [os.path.join(pclass, vid) for vid in videos]) )

    #return np.array(dataset)
    return sorted(classlen.items(), key=operator.itemgetter(1), reverse=True)


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

def index_spread(sequence, n):
    length = float(len(sequence))
    #s = set()
    #for i in range(n):
        #s.add(int(ceil(i * length / n)))
    return np.linspace(0, length-1, num=n, dtype=int)

def getOpenfaceData(lfwAligned, nPpl, nImgs, imgSize):
    X, y = [], [] # X are now list of video frame lists

    personNum = 0
    for (person, numVideos) in lfwAligned[:nPpl]:
        vids = np.array(sorted(os.listdir(person)))
        for vid in vids[:4]:
            vidDir = os.path.join(person, vid)
            if os.path.isdir(vidDir):
                imgs = np.array(sorted(os.listdir(vidDir)))
                if len(imgs) > 0:
                    images = []
                    for imgP in imgs[index_spread(imgs, nImgs)]: # TODO choose frame metric
                        imgPath = os.path.join(vidDir, imgP)
                        #print imgPath
                        img = cv2.imread(imgPath)
                        img = cv2.resize(img, (imgSize, imgSize))
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                        images.append(img)
                    X.append(images)
                    y.append(personNum)

        personNum += 1

    X = np.array(X)
    y = np.array(y)
    return (X, y)


def getFacenetData(lfwAligned, nPpl, nImgs, imgSize):
    X, y = [], []

    personNum = 0
    for (person, numVideos) in lfwAligned[:nPpl]:
        vids = np.array(sorted(os.listdir(person)))
        for vid in vids[:4]:
        vidDir = os.path.join(person, vid)
            if os.path.isdir(vidDir):
                imgs = np.array(sorted(os.listdir(vidDir)))
                if len(imgs) > 0:
                    images = []
                    for imgP in imgs[index_spread(imgs, nImgs)]: # TODO choose frame metric
                        imgPath = os.path.join(vidDir, imgP)
                        img = misc.imread(imgPath)

                        if img.ndim == 2:
                            img = to_rgb(img)
                        img = prewhiten(img)
                        img = crop(img, False, imgSize)
                        images.append(img)

                    X.append(images)
                    y.append(personNum)

        personNum += 1

    X = np.array(X)
    y = np.array(y)
    return (X, y)

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


def openfaceExp(lfwAligned, net, cls):
    df = pd.DataFrame(columns=('nPpl', 'nImgs',
                               'trainTimeSecMean', 'trainTimeSecStd',
                               'predictTimeSecMean', 'predictTimeSecStd',
                               'accsMean', 'accsStd'))

    repCache = {}

    df_i = 0
    for nPpl in nPplVals:
        print(" + nPpl: {}".format(nPpl))
        (X, y) = getOpenfaceData(lfwAligned, nPpl, nImgs, 96) # picks from sorted persons, first video and nImgs frames evenly spread
        nSampled = X.shape[0]
        print 'nSampled:', nSampled
        ss = ShuffleSplit(nSampled, n_iter=10, test_size=0.1, random_state=0) # we must split videos not images here, so training and test is not done on same video!

        allTrainTimeSec = []
        allPredictTimeSec = []
        accs = []

        for train, test in ss:
        #print 'split:', train, test
            X_train = []
            Y_train = []
            for index, vid in zip(train, X[train]):
                rep_array = []
                for img in vid:
                    h = hash(str(img.data))
                    if h in repCache:
                        rep = repCache[h]
                    else:
                        rep = net.forward(img)
                        repCache[h] = rep
                    rep_array.append(rep)
                rep_array = np.array(rep_array)
                print 'train', rep_array.shape, rep_array.mean(axis=0).shape
                X_train.append(rep_array.mean(axis=0))
                Y_train.append(y[index])

            start = time.time()
            #print 'reps:', X_train,'\n',Y_train
            X_train = np.array(X_train)
            Y_train = np.array(Y_train)
            cls.fit(X_train, Y_train)
            trainTimeSec = time.time() - start
            allTrainTimeSec.append(trainTimeSec)

            start = time.time()
            X_test = []
            Y_test = []
            for index, vid in zip(test, X[test]):
                rep_array = []
                for img in vid:
                    rep_array.append(net.forward(img))
                rep_array = np.array(rep_array)
                print 'test', rep_array.shape, rep_array.mean(axis=0).shape
                X_test.append(rep_array.mean(axis=0))
                Y_test.append(y[index])

            X_test = np.array(X_test)
            y_predict = cls.predict(X_test)

            predictTimeSec = time.time() - start
            allPredictTimeSec.append(predictTimeSec / len(test))
            y_predict = np.array(y_predict)
            Y_test = np.array(Y_test)
            acc = accuracy_score(Y_test, y_predict)
            accs.append(acc)

        print 'acc:', accs
        df.loc[df_i] = [nPpl, nImgs,
                        np.mean(allTrainTimeSec), np.std(allTrainTimeSec),
                        np.mean(allPredictTimeSec), np.std(allPredictTimeSec),
                        np.mean(accs), np.std(accs)]
        df_i += 1

    return df


def facenetExp(lfwAligned, facenetModelDir, cls):
    df = pd.DataFrame(columns=('nPpl', 'nImgs',
                               'trainTimeSecMean', 'trainTimeSecStd',
                               'predictTimeSecMean', 'predictTimeSecStd',
                               'accsMean', 'accsStd'))


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

            repCache = {}

            df_i = 0
            for nPpl in nPplVals:
                print(" + nPpl: {}".format(nPpl))
                (X, y) = getFacenetData(lfwAligned, nPpl, nImgs, image_size)
                nSampled = X.shape[0]
                print 'nSampled:', nSampled
                ss = ShuffleSplit(nSampled, n_iter=10, test_size=0.1, random_state=0)

                allTrainTimeSec = []
                allPredictTimeSec = []
                accs = []

                for train, test in ss:
                    #print 'split:', train, test
                    X_train = []
                    Y_train = []
                    for index, vid in zip(train, X[train]):
                        imgs = vid # use vid as batch and one forward
                        imgs = np.array(imgs)
                        feed_dict = { images_placeholder:imgs }
                        rep_array = sess.run(embeddings, feed_dict=feed_dict)
                        rep_array = np.array(rep_array)
                        print 'train', rep_array.shape, rep_array.mean(axis=0).shape
                        X_train.append(rep_array.mean(axis=0))
                        Y_train.append(y[index])

                    start = time.time()
                    X_train = np.array(X_train)
                    Y_train = np.array(Y_train)
                    cls.fit(X_train, Y_train)
                    trainTimeSec = time.time() - start
                    allTrainTimeSec.append(trainTimeSec)

                    start = time.time()
                    X_test = []
                    Y_test = []
                    for index, vid in zip(test, X[test]):
                        imgs = vid
                        imgs = np.array(imgs)
                        feed_dict = { images_placeholder:imgs }
                        rep_array = sess.run(embeddings, feed_dict=feed_dict)
                        rep_array = np.array(rep_array)
                        print 'test', rep_array.shape, rep_array.mean(axis=0).shape 
                        X_test.append(rep_array.mean(axis=0))
                        Y_test.append(y[index])

                    y_predict = cls.predict(X_test)
                    predictTimeSec = time.time() - start
                    allPredictTimeSec.append(predictTimeSec / len(test))
                    y_predict = np.array(y_predict)
                    Y_test = np.array(Y_test)
                    acc = accuracy_score(Y_test, y_predict)
                    accs.append(acc)
                
                print 'accs:', accs
                df.loc[df_i] = [nPpl, nImgs,
                                np.mean(allTrainTimeSec), np.std(allTrainTimeSec),
                                np.mean(allPredictTimeSec), np.std(allPredictTimeSec),
                                np.mean(accs), np.std(accs)]
                df_i += 1

    return df


def plotAccuracy(workDir, largeFont, openfaceCPUsvmDf, openfaceCPUlinearsvmDf, facenetsvmDf, facenetlinearsvmDf):
    indices = openfaceCPUsvmDf.index.values
    barWidth = 0.15

    if largeFont:
        fig = plt.figure(figsize=(10, 5))
    else:
        fig = plt.figure(figsize=(14, 4))
    ax = fig.add_subplot(111)
    plt.bar(indices, openfaceCPUsvmDf['accsMean'], barWidth,
            yerr=openfaceCPUsvmDf['accsStd'], label='OpenFace',
            color=colors[0], ecolor='0.3', alpha=alpha)
    plt.bar(indices + 1 * barWidth, openfaceCPUlinearsvmDf['accsMean'], barWidth,
            yerr=openfaceCPUlinearsvmDf['accsStd'], label='OpenFace LinearSVC',
            color=colors[1], ecolor='0.3', alpha=alpha)
    plt.bar(indices + 2 * barWidth, facenetsvmDf['accsMean'], barWidth,
            yerr=facenetsvmDf['accsStd'], label='Facenet SVM',
            color=colors[2], ecolor='0.3', alpha=alpha)
    plt.bar(indices + 3 * barWidth, facenetlinearsvmDf['accsMean'], barWidth,
            yerr=facenetlinearsvmDf['accsStd'], label='Facenet LinearSVC',
            color=colors[3], ecolor='0.3', alpha=alpha)

    box = ax.get_position()
    if largeFont:
        ax.set_position([box.x0, box.y0 + 0.07, box.width, box.height * 0.83])
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=6,
                   fancybox=True, shadow=True, fontsize=16)
    else:
        ax.set_position([box.x0, box.y0 + 0.05, box.width, box.height * 0.85])
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=6,
                   fancybox=True, shadow=True)
    plt.ylabel("Classification Accuracy")
    plt.xlabel("Number of People")

    ax.set_xticks(indices + 2.5 * barWidth)
    xticks = []
    for nPpl in nPplVals:
        xticks.append(nPpl)
    ax.set_xticklabels(xticks)

    locs, labels = plt.xticks()
    plt.ylim(0, 1)
    plt.savefig(os.path.join(workDir, 'accuracies.png'))

    with open(os.path.join(workDir, 'accuracies.txt'), "w") as f:
        f.write('OpenFace SVM: {1}, {0}\n'.format(openfaceCPUsvmDf['accsStd'], openfaceCPUsvmDf['accsMean']))
        f.write('OpenFace LinearSVC: {1}, {0}\n'.format(openfaceCPUlinearsvmDf['accsStd'], openfaceCPUlinearsvmDf['accsMean']))
        f.write('Facenet LinearSVC {1}, {0}\n'.format(facenetlinearsvmDf['accsStd'], facenetlinearsvmDf['accsMean']))
        f.write('Facenet SVM {1}, {0}\n'.format(facenetsvmDf['accsStd'], facenetsvmDf['accsMean']))


def plotTrainingTime(workDir, largeFont, openfaceCPUsvmDf, openfaceCPUlinearsvmDf, facenetsvmDf, facenetlinearsvmDf):
    indices = openfaceCPUsvmDf.index.values
    barWidth = 0.15

    fig = plt.figure(figsize=(14, 4))
    ax = fig.add_subplot(111)
    plt.bar(indices, openfaceCPUsvmDf['trainTimeSecMean'], barWidth,
            yerr=openfaceCPUsvmDf['trainTimeSecStd'],
            label='OpenFace SVM',
            color=colors[0], ecolor='0.3', alpha=alpha)
    plt.bar(indices + 1 * barWidth, openfaceCPUlinearsvmDf['trainTimeSecMean'], barWidth,
            yerr=openfaceCPUlinearsvmDf['trainTimeSecStd'],
            label='OpenFace LinearSVC',
            color=colors[1], ecolor='0.3', alpha=alpha)
    plt.bar(indices + 2 * barWidth, facenetsvmDf['trainTimeSecMean'], barWidth,
            yerr=facenetsvmDf['trainTimeSecStd'],
            label='Facenet SVM',
            color=colors[2], ecolor='0.3', alpha=alpha)
    plt.bar(indices + 3 * barWidth, facenetlinearsvmDf['trainTimeSecMean'], barWidth,
            yerr=facenetlinearsvmDf['trainTimeSecStd'],
            label='Facenet LinearSVC',
            color=colors[3], ecolor='0.3', alpha=alpha)

    box = ax.get_position()
    if largeFont:
        ax.set_position([box.x0, box.y0 + 0.08, box.width, box.height * 0.83])
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.27), ncol=6,
                   fancybox=True, shadow=True, fontsize=16)
    else:
        ax.set_position([box.x0, box.y0 + 0.05, box.width, box.height * 0.85])
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=6,
                   fancybox=True, shadow=True)
    plt.ylabel("Training Time (s)")
    plt.xlabel("Number of People")

    ax.set_xticks(indices + 2.5 * barWidth)
    xticks = []
    for nPpl in nPplVals:
        xticks.append(nPpl)
    ax.set_xticklabels(xticks)
    locs, labels = plt.xticks()
    # plt.setp(labels, rotation=45)
    # plt.ylim(0, 1)
    ax.set_yscale('log')
    plt.savefig(os.path.join(workDir, 'trainTimes.png'))

    with open(os.path.join(workDir, 'trainTimes.txt'), "w") as f:
        f.write('OpenFace SVM: {1}, {0}\n'.format(openfaceCPUsvmDf['trainTimeSecStd'], openfaceCPUsvmDf['trainTimeSecMean']))
        f.write('OpenFace LinearSVC: {1}, {0}\n'.format(openfaceCPUlinearsvmDf['trainTimeSecStd'], openfaceCPUlinearsvmDf['trainTimeSecMean']))
        f.write('Facenet LinearSVC: {1}, {0}\n'.format(facenetlinearsvmDf['trainTimeSecStd'], facenetlinearsvmDf['trainTimeSecMean']))
        f.write('Facenet SVM: {1}, {0}\n'.format(facenetsvmDf['trainTimeSecStd'], facenetsvmDf['trainTimeSecMean']))


def plotPredictionTime(workDir, largeFont, openfaceCPUsvmDf, openfaceCPUlinearsvmDf, facenetsvmDf, facenetlinearsvmDf):
    indices = openfaceCPUsvmDf.index.values
    barWidth = 0.15

    fig = plt.figure(figsize=(14, 4))
    ax = fig.add_subplot(111)
    plt.bar(indices, openfaceCPUsvmDf['predictTimeSecMean'], barWidth,
            yerr=openfaceCPUsvmDf['predictTimeSecStd'],
            label='OpenFace SVM',
            color=colors[0], ecolor='0.3', alpha=alpha)
    plt.bar(indices + 1 * barWidth, openfaceCPUlinearsvmDf['predictTimeSecMean'], barWidth,
            yerr=openfaceCPUlinearsvmDf['predictTimeSecStd'],
            label='OpenFace LinearSVC',
            color=colors[1], ecolor='0.3', alpha=alpha)
    plt.bar(indices + 2 * barWidth, facenetsvmDf['predictTimeSecMean'], barWidth,
            yerr=facenetsvmDf['predictTimeSecStd'],
            label='Facenet SVM',
            color=colors[2], ecolor='0.3', alpha=alpha)
    plt.bar(indices + 3 * barWidth, facenetlinearsvmDf['predictTimeSecMean'], barWidth,
            yerr=facenetlinearsvmDf['predictTimeSecStd'],
            label='Facenet LinearSVC',
            color=colors[3], ecolor='0.3', alpha=alpha)


    box = ax.get_position()
    if largeFont:
        ax.set_position([box.x0, box.y0 + 0.11, box.width, box.height * 0.7])
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.45), ncol=6,
                   fancybox=True, shadow=True, fontsize=16)
    else:
        ax.set_position([box.x0, box.y0 + 0.05, box.width, box.height * 0.77])
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.37), ncol=6,
                   fancybox=True, shadow=True)
    plt.ylabel("Prediction Time (s)")
    plt.xlabel("Number of People")
    ax.set_xticks(indices + 2.5 * barWidth)
    xticks = []
    for nPpl in nPplVals:
        xticks.append(nPpl)
    ax.set_xticklabels(xticks)
    ax.xaxis.grid(False)
    locs, labels = plt.xticks()
    # plt.setp(labels, rotation=45)
    # plt.ylim(0, 1)

    ax.set_yscale('log')
    plt.savefig(os.path.join(workDir, 'predictTimes.png'))

    with open(os.path.join(workDir, 'predictTimes.txt'), "w") as f:
        f.write('OpenFace SVM: {1}, {0}\n'.format(openfaceCPUsvmDf['predictTimeSecStd'], openfaceCPUsvmDf['predictTimeSecMean']))
        f.write('OpenFace LinearSVC: {1}, {0}\n'.format(openfaceCPUlinearsvmDf['predictTimeSecStd'], openfaceCPUlinearsvmDf['predictTimeSecMean']))
        f.write('Facenet LinearSVC: {1}, {0}\n'.format(facenetlinearsvmDf['predictTimeSecStd'], facenetlinearsvmDf['predictTimeSecMean']))
        f.write('Facenet SVM: {1}, {0}\n'.format(facenetsvmDf['predictTimeSecStd'], facenetsvmDf['predictTimeSecMean']))


if __name__ == '__main__':
    main()

