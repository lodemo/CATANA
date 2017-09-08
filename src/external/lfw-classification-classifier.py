#!/usr/bin/env python2
#
# Copyright 2015-2016 Carnegie Mellon University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This file implements a non-standard LFW classification experiment for
# the purposes of benchmarking the performance and accuracies of
# classification techniques.
# For the standard LFW experiment, see lfw.py.

import cv2
import numpy as np
import pandas as pd

from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import ShuffleSplit
from sklearn.metrics import accuracy_score
from sklearn.mixture import GMM
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

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
from threading import Thread

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
facenetModelDir = os.path.join(facenetDir, 'models', '20170117-215115')
#facenetModelDir = os.path.join(facenetDir, 'models', '20161116-234200')

nPplVals = [100, 400, 600]
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

    print("Getting lfwPpl")
    lfwPplCache = os.path.join(args.workDir, 'lfwPpl.pkl')
    lfwPpl = cacheToFile(lfwPplCache)(getLfwPplSorted)(args.lfwDlibAligned)

    print("Getting lfwMtcnnPpl")
    lfwMtcnnPplCache = os.path.join(args.workDir, 'lfwMtcnnPpl.pkl')
    lfwMtcnnPpl = cacheToFile(lfwMtcnnPplCache)(getLfwPplSorted)(args.lfwMtcnnAligned)


    print("OpenFace SVM Experiment")
    net = openface.TorchNeuralNet(openfaceModelPath, 96, cuda=False)
    cls = SVC(C=1, kernel='linear')
    cache = os.path.join(args.workDir, 'openface.cpu.svm.pkl')
    openfaceCPUsvmDf = cacheToFile(cache)(openfaceExp)(lfwPpl, net, cls)

    print("OpenFace LinearSVC Experiment")
    net = openface.TorchNeuralNet(openfaceModelPath, 96, cuda=False)
    cls = LinearSVC(C=1, multi_class='ovr')
    cache = os.path.join(args.workDir, 'openface.cpu.linearsvm.pkl')
    openfaceCPUlinearsvmDf = cacheToFile(cache)(openfaceExp)(lfwPpl, net, cls)

    print("OpenFace GaussianNB Experiment")
    net = openface.TorchNeuralNet(openfaceModelPath, 96, cuda=False)
    cls = GaussianNB()
    cache = os.path.join(args.workDir, 'openface.cpu.gaussiannb.pkl')
    openfacegaussiannbDf = cacheToFile(cache)(openfaceExp)(lfwPpl, net, cls)

    print("OpenFace DecisionTree Experiment")
    net = openface.TorchNeuralNet(openfaceModelPath, 96, cuda=False)
    cls = DecisionTreeClassifier(max_depth=20)
    cache = os.path.join(args.workDir, 'openface.cpu.dectree.pkl')
    openfacedectreeDf = cacheToFile(cache)(openfaceExp)(lfwPpl, net, cls)

    print("OpenFace DecisionTree Experiment")
    net = openface.TorchNeuralNet(openfaceModelPath, 96, cuda=False)
    cls = LogisticRegression(C=1, multi_class ='ovr')
    cache = os.path.join(args.workDir, 'openface.cpu.logreg.pkl')
    openfacelogregDf = cacheToFile(cache)(openfaceExp)(lfwPpl, net, cls)


    print("Facenet SVM Experiment")
    cls = SVC(C=1, kernel='linear')
    cache = os.path.join(args.workDir, 'facenet.svm.pkl')
    facenetsvmDf = cacheToFile(cache)(facenetExp)(lfwMtcnnPpl, facenetModelDir, cls)

    print("Facenet LinearSVC Experiment")
    cls = LinearSVC(C=1, multi_class='ovr')
    cache = os.path.join(args.workDir, 'facenet.linearsvm.pkl')
    facenetlinearsvmDf = cacheToFile(cache)(facenetExp)(lfwMtcnnPpl, facenetModelDir, cls)

    print("Facenet GaussianNB Experiment")
    cls = GaussianNB()
    cache = os.path.join(args.workDir, 'facenet.gaussiannb.pkl')
    facenetgaussiannbDf = cacheToFile(cache)(facenetExp)(lfwMtcnnPpl, facenetModelDir, cls)

    print("Facenet DecisionTree Experiment")
    cls = DecisionTreeClassifier(max_depth=20)
    cache = os.path.join(args.workDir, 'facenet.dectree.pkl')
    facenetdectreeDf = cacheToFile(cache)(facenetExp)(lfwMtcnnPpl, facenetModelDir, cls)

    print("Facenet LogisticRegression Experiment")
    cls = LogisticRegression(C=1, multi_class ='ovr')
    cache = os.path.join(args.workDir, 'facenet.logreg.pkl')
    facenetlogregDf = cacheToFile(cache)(facenetExp)(lfwMtcnnPpl, facenetModelDir, cls)

    plotAccuracy(args.workDir, args.largeFont,
                    openfaceCPUsvmDf, openfaceCPUlinearsvmDf, openfacegaussiannbDf,
                    openfacedectreeDf, openfacelogregDf, facenetsvmDf, facenetlinearsvmDf, facenetgaussiannbDf, facenetdectreeDf, facenetlogregDf)
    plotTrainingTime(args.workDir, args.largeFont,
                    openfaceCPUsvmDf, openfaceCPUlinearsvmDf, openfacegaussiannbDf,
                    openfacedectreeDf, openfacelogregDf, facenetsvmDf, facenetlinearsvmDf, facenetgaussiannbDf, facenetdectreeDf, facenetlogregDf)
    plotPredictionTime(args.workDir, args.largeFont,
                    openfaceCPUsvmDf, openfaceCPUlinearsvmDf, openfacegaussiannbDf,
                    openfacedectreeDf, openfacelogregDf, facenetsvmDf, facenetlinearsvmDf, facenetgaussiannbDf, facenetdectreeDf, facenetlogregDf)

# http://stackoverflow.com/questions/16463582


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


def getLfwPplSorted(lfwAligned):
    lfwPpl = {}
    for person in os.listdir(lfwAligned):
        fullPath = os.path.join(lfwAligned, person)
        if os.path.isdir(fullPath):
            nFiles = len([item for item in os.listdir(fullPath)
                          if os.path.isfile(os.path.join(fullPath, item))])
            lfwPpl[fullPath] = nFiles

    df = pd.DataFrame({'len': lfwPpl})
    print 'Lfw num images:\n', df.describe()

    return sorted(lfwPpl.items(), key=operator.itemgetter(1), reverse=True)


def getData(lfwPpl, nPpl, nImgs, size, mode):
    X, y = [], []

    personNum = 0
    for (person, nTotalImgs) in lfwPpl[:nPpl]:
        imgs = sorted(os.listdir(person))
        for imgPath in imgs[:nImgs]:
            imgPath = os.path.join(person, imgPath)
            img = cv2.imread(imgPath)
            img = cv2.resize(img, (size, size))
            if mode == 'grayscale':
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            elif mode == 'rgb':
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                assert 0

            X.append(img)
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

def getFacenetData(lfwPpl, nPpl, nImgs, imgSize, do_prewhiten=True):
    X, y = [], []

    personNum = 0
    for (person, nTotalImgs) in lfwPpl[:nPpl]:
        imgs = sorted(os.listdir(person))
        for imgPath in imgs[:nImgs]:
            imgPath = os.path.join(person, imgPath)
            img = misc.imread(imgPath)

            if img.ndim == 2:
                img = to_rgb(img)
            if do_prewhiten:
                img = prewhiten(img)
            img = crop(img, False, imgSize)

            X.append(img)
            y.append(personNum)

        personNum += 1

    X = np.array(X)
    y = np.array(y)
    return (X, y)


def opencvExp(lfwAligned, cls):
    df = pd.DataFrame(columns=('nPpl', 'nImgs', 'trainTimeSecMean', 'trainTimeSecStd',
                               'predictTimeSecMean', 'predictTimeSecStd',
                               'accsMean', 'accsStd'))

    df_i = 0
    for nPpl in nPplVals:
        print(" + nPpl: {}".format(nPpl))
        (X, y) = getData(lfwAligned, nPpl, nImgs, size=96, mode='grayscale')
        nSampled = X.shape[0]
        ss = ShuffleSplit(nSampled, n_iter=10, test_size=0.1, random_state=0)

        allTrainTimeSec = []
        allPredictTimeSec = []
        accs = []

        for train, test in ss:
            start = time.time()
            cls.train(X[train], y[train])
            trainTimeSec = time.time() - start
            allTrainTimeSec.append(trainTimeSec)

            y_predict = []
            for img in X[test]:
                start = time.time()
                (label, score) = cls.predict(img)
                y_predict.append(label)
                predictTimeSec = time.time() - start
                allPredictTimeSec.append(predictTimeSec)
            y_predict = np.array(y_predict)

            acc = accuracy_score(y[test], y_predict)
            accs.append(acc)

        df.loc[df_i] = [nPpl, nImgs,
                        np.mean(allTrainTimeSec), np.std(allTrainTimeSec),
                        np.mean(allPredictTimeSec), np.std(allPredictTimeSec),
                        np.mean(accs), np.std(accs)]
        df_i += 1

    return df


def openfaceExp(lfwAligned, net, cls):
    df = pd.DataFrame(columns=('nPpl', 'nImgs',
                               'trainTimeSecMean', 'trainTimeSecStd',
                               'predictTimeSecMean', 'predictTimeSecStd',
                               'accsMean', 'accsStd'))

    repCache = {}

    df_i = 0
    for nPpl in nPplVals:
        print(" + nPpl: {}".format(nPpl))
        (X, y) = getData(lfwAligned, nPpl, nImgs, size=96, mode='rgb')
        nSampled = X.shape[0]
        ss = ShuffleSplit(nSampled, n_iter=10, test_size=0.1, random_state=0)

        allTrainTimeSec = []
        allPredictTimeSec = []
        accs = []

        for train, test in ss:
            X_train = []
            for img in X[train]:
                h = hash(str(img.data))
                if h in repCache:
                    rep = repCache[h]
                else:
                    rep = net.forward(img)
                    repCache[h] = rep
                X_train.append(rep)

            start = time.time()
            X_train = np.array(X_train)
            cls.fit(X_train, y[train])
            trainTimeSec = time.time() - start
            allTrainTimeSec.append(trainTimeSec)

            start = time.time()
            X_test = []
            for img in X[test]:
                X_test.append(net.forward(img))
            y_predict = cls.predict(X_test)
            predictTimeSec = time.time() - start
            allPredictTimeSec.append(predictTimeSec / len(test))
            y_predict = np.array(y_predict)

            acc = accuracy_score(y[test], y_predict)
            accs.append(acc)

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
                ss = ShuffleSplit(nSampled, n_iter=10, test_size=0.1, random_state=0)

                allTrainTimeSec = []
                allPredictTimeSec = []
                accs = []

                for train, test in ss:
                    X_train = []
                    for img in X[train]:
                        h = hash(str(img.data))
                        if h in repCache:
                            rep = repCache[h]
                        else:
                            imgs = [img]
                            imgs = np.array(imgs)
                            feed_dict = { images_placeholder:imgs }
                            emb = sess.run(embeddings, feed_dict=feed_dict)
                            rep = emb[0]
                            repCache[h] = rep
                        X_train.append(rep)

                    start = time.time()
                    X_train = np.array(X_train)
                    cls.fit(X_train, y[train])
                    trainTimeSec = time.time() - start
                    allTrainTimeSec.append(trainTimeSec)

                    start = time.time()
                    X_test = []
                    for img in X[test]:
                        imgs = [img]
                        imgs = np.array(imgs)
                        feed_dict = { images_placeholder:imgs }
                        emb = sess.run(embeddings, feed_dict=feed_dict)
                        X_test.append(emb[0])
                    y_predict = cls.predict(X_test)
                    predictTimeSec = time.time() - start
                    allPredictTimeSec.append(predictTimeSec / len(test))
                    y_predict = np.array(y_predict)

                    acc = accuracy_score(y[test], y_predict)
                    accs.append(acc)

                df.loc[df_i] = [nPpl, nImgs,
                                np.mean(allTrainTimeSec), np.std(allTrainTimeSec),
                                np.mean(allPredictTimeSec), np.std(allPredictTimeSec),
                                np.mean(accs), np.std(accs)]
                df_i += 1

    return df


def plotAccuracy(workDir, largeFont, openfaceCPUsvmDf, openfaceCPUlinearsvmDf, openfacegaussiannbDf,
                    openfacedectreeDf, openfacelogregDf, facenetsvmDf, facenetlinearsvmDf, facenetgaussiannbDf, facenetdectreeDf, facenetlogregDf):
    indices = openfaceCPUsvmDf.index.values
    barWidth = 0.08

    if largeFont:
        fig = plt.figure(figsize=(20, 5))
    else:
        fig = plt.figure(figsize=(22, 4))
    ax = fig.add_subplot(111)
    plt.bar(indices, openfaceCPUsvmDf['accsMean'], barWidth,
            yerr=openfaceCPUsvmDf['accsStd'], label='OpenFace SVM',
            color=colors[0], ecolor='0.3', alpha=alpha)
    plt.bar(indices + barWidth, openfaceCPUlinearsvmDf['accsMean'], barWidth,
            yerr=openfaceCPUlinearsvmDf['accsStd'], label='OpenFace LinearSVC',
            color=colors[1], ecolor='0.3', alpha=alpha)
    plt.bar(indices + 2 * barWidth, openfacegaussiannbDf['accsMean'], barWidth,
            yerr=openfacegaussiannbDf['accsStd'], label='OpenFace GaussianNB',
            color=colors[2], ecolor='0.3', alpha=alpha)
    plt.bar(indices + 3 * barWidth, openfacedectreeDf['accsMean'], barWidth,
            yerr=openfacedectreeDf['accsStd'], label='OpenFace DecisionTree',
            color=colors[3], ecolor='0.3', alpha=alpha)
    plt.bar(indices + 4 * barWidth, openfacelogregDf['accsMean'], barWidth,
            yerr=openfacelogregDf['accsStd'], label='OpenFace LogisticRegression',
            color=colors[4], ecolor='0.3', alpha=alpha)
    plt.bar(indices + 5 * barWidth, facenetsvmDf['accsMean'], barWidth,
            yerr=facenetsvmDf['accsStd'], label='Facenet SVM',
            color=colors[5], ecolor='0.3', alpha=alpha)
    plt.bar(indices + 6 * barWidth, facenetlinearsvmDf['accsMean'], barWidth,
            yerr=facenetlinearsvmDf['accsStd'], label='Facenet LinearSVC',
            color=colors[6], ecolor='0.3', alpha=alpha)
    plt.bar(indices + 7 * barWidth, facenetgaussiannbDf['accsMean'], barWidth,
            yerr=facenetgaussiannbDf['accsStd'], label='Facenet GaussianNB',
            color=colors[7], ecolor='0.3', alpha=alpha)
    plt.bar(indices + 8 * barWidth, facenetdectreeDf['accsMean'], barWidth,
            yerr=facenetdectreeDf['accsStd'], label='Facenet DecisionTree',
            color=colors[8], ecolor='0.3', alpha=alpha)
    plt.bar(indices + 9 * barWidth, facenetlogregDf['accsMean'], barWidth,
            yerr=facenetlogregDf['accsStd'], label='Facenet LogisticRegression',
            color=colors[9], ecolor='0.3', alpha=alpha)

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

    ax.set_xticks(indices + 4 * barWidth)
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
        f.write('OpenFace GaussianNB: {1}, {0}\n'.format(openfacegaussiannbDf['accsStd'], openfacegaussiannbDf['accsMean']))
        f.write('OpenFace DecisionTree: {1}, {0}\n'.format(openfacedectreeDf['accsStd'], openfacedectreeDf['accsMean']))
        f.write('OpenFace LogisticRegression: {1}, {0}\n'.format(openfacelogregDf['accsStd'], openfacelogregDf['accsMean']))
        f.write('Facenet SVM: {1}, {0}\n'.format(facenetsvmDf['accsStd'], facenetsvmDf['accsMean']))
        f.write('Facenet LinearSVC {1}, {0}\n'.format(facenetlinearsvmDf['accsStd'], facenetlinearsvmDf['accsMean']))
        f.write('Facenet GaussianNB {1}, {0}\n'.format(facenetgaussiannbDf['accsStd'], facenetgaussiannbDf['accsMean']))
        f.write('Facenet DecisionTree {1}, {0}\n'.format(facenetdectreeDf['accsStd'], facenetdectreeDf['accsMean']))
        f.write('Facenet LogisticRegression {1}, {0}\n'.format(facenetlogregDf['accsStd'], facenetlogregDf['accsMean']))

def plotTrainingTime(workDir, largeFont, openfaceCPUsvmDf, openfaceCPUlinearsvmDf, openfacegaussiannbDf,
                    openfacedectreeDf, openfacelogregDf, facenetsvmDf, facenetlinearsvmDf, facenetgaussiannbDf, facenetdectreeDf, facenetlogregDf):
    indices = openfaceCPUsvmDf.index.values
    barWidth = 0.08

    fig = plt.figure(figsize=(22, 4))
    ax = fig.add_subplot(111)
    plt.bar(indices, openfaceCPUsvmDf['trainTimeSecMean'], barWidth,
            yerr=openfaceCPUsvmDf['trainTimeSecStd'], label='OpenFace SVM',
            color=colors[0], ecolor='0.3', alpha=alpha)
    plt.bar(indices + barWidth, openfaceCPUlinearsvmDf['trainTimeSecMean'], barWidth,
            yerr=openfaceCPUlinearsvmDf['trainTimeSecStd'], label='OpenFace LinearSVC',
            color=colors[1], ecolor='0.3', alpha=alpha)
    plt.bar(indices + 2 * barWidth, openfacegaussiannbDf['trainTimeSecMean'], barWidth,
            yerr=openfacegaussiannbDf['trainTimeSecStd'], label='OpenFace GaussianNB',
            color=colors[2], ecolor='0.3', alpha=alpha)
    plt.bar(indices + 3 * barWidth, openfacedectreeDf['trainTimeSecMean'], barWidth,
            yerr=openfacedectreeDf['trainTimeSecStd'],
            label='OpenFace DecisionTree',
            color=colors[3], ecolor='0.3', alpha=alpha)
    plt.bar(indices + 4 * barWidth, openfacelogregDf['trainTimeSecMean'], barWidth,
            yerr=openfacelogregDf['trainTimeSecStd'],
            label='OpenFace LogisticRegression',
            color=colors[4], ecolor='0.3', alpha=alpha)
    plt.bar(indices + 5 * barWidth, facenetsvmDf['trainTimeSecMean'], barWidth,
            yerr=facenetsvmDf['trainTimeSecStd'],
            label='Facenet SVM',
            color=colors[5], ecolor='0.3', alpha=alpha)
    plt.bar(indices + 6 * barWidth, facenetlinearsvmDf['trainTimeSecMean'], barWidth,
            yerr=facenetlinearsvmDf['trainTimeSecStd'],
            label='Facenet LinearSVC',
            color=colors[6], ecolor='0.3', alpha=alpha)
    plt.bar(indices + 7 * barWidth, facenetgaussiannbDf['trainTimeSecMean'], barWidth,
            yerr=facenetgaussiannbDf['trainTimeSecStd'],
            label='Facenet GaussianNB',
            color=colors[7], ecolor='0.3', alpha=alpha)
    plt.bar(indices + 8 * barWidth, facenetdectreeDf['trainTimeSecMean'], barWidth,
            yerr=facenetdectreeDf['trainTimeSecStd'],
            label='Facenet DecisionTree',
            color=colors[8], ecolor='0.3', alpha=alpha)
    plt.bar(indices + 9 * barWidth, facenetlogregDf['trainTimeSecMean'], barWidth,
            yerr=facenetlogregDf['trainTimeSecStd'],
            label='Facenet LogisticRegression',
            color=colors[9], ecolor='0.3', alpha=alpha)

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

    ax.set_xticks(indices + 4 * barWidth)
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
        f.write('OpenFace GaussianNB: {1}, {0}\n'.format(openfacegaussiannbDf['trainTimeSecStd'], openfacegaussiannbDf['trainTimeSecMean']))
        f.write('OpenFace DecisionTree: {1}, {0}\n'.format(openfacedectreeDf['trainTimeSecStd'], openfacedectreeDf['trainTimeSecMean']))
        f.write('OpenFace LogisticRegression: {1}, {0}\n'.format(openfacelogregDf['trainTimeSecStd'], openfacelogregDf['trainTimeSecMean']))
        f.write('Facenet SVM: {1}, {0}\n'.format(facenetsvmDf['trainTimeSecStd'], facenetsvmDf['trainTimeSecMean']))
        f.write('Facenet LinearSVC {1}, {0}\n'.format(facenetlinearsvmDf['trainTimeSecStd'], facenetlinearsvmDf['trainTimeSecMean']))
        f.write('Facenet GaussianNB {1}, {0}\n'.format(facenetgaussiannbDf['trainTimeSecStd'], facenetgaussiannbDf['trainTimeSecMean']))
        f.write('Facenet DecisionTree {1}, {0}\n'.format(facenetdectreeDf['trainTimeSecStd'], facenetdectreeDf['trainTimeSecMean']))
        f.write('Facenet LogisticRegression {1}, {0}\n'.format(facenetlogregDf['trainTimeSecStd'], facenetlogregDf['trainTimeSecMean']))

def plotPredictionTime(workDir, largeFont, openfaceCPUsvmDf, openfaceCPUlinearsvmDf, openfacegaussiannbDf,
                    openfacedectreeDf, openfacelogregDf, facenetsvmDf, facenetlinearsvmDf, facenetgaussiannbDf, facenetdectreeDf, facenetlogregDf):
    indices = openfaceCPUsvmDf.index.values
    barWidth = 0.08

    fig = plt.figure(figsize=(22, 4))
    ax = fig.add_subplot(111)
    plt.bar(indices, openfaceCPUsvmDf['predictTimeSecMean'], barWidth,
            yerr=openfaceCPUsvmDf['predictTimeSecStd'], label='OpenFace SVM',
            color=colors[0], ecolor='0.3', alpha=alpha)
    plt.bar(indices + barWidth, openfaceCPUlinearsvmDf['predictTimeSecMean'], barWidth,
            yerr=openfaceCPUlinearsvmDf['predictTimeSecStd'], label='OpenFace LinearSVC',
            color=colors[1], ecolor='0.3', alpha=alpha)
    plt.bar(indices + 2 * barWidth, openfacegaussiannbDf['predictTimeSecMean'], barWidth,
            yerr=openfacegaussiannbDf['predictTimeSecStd'], label='OpenFace GaussianNB',
            color=colors[2], ecolor='0.3', alpha=alpha)
    plt.bar(indices + 3 * barWidth, openfacedectreeDf['predictTimeSecMean'], barWidth,
            yerr=openfacedectreeDf['predictTimeSecStd'],
            label='OpenFace DecisionTree',
            color=colors[3], ecolor='0.3', alpha=alpha)
    plt.bar(indices + 4 * barWidth, openfacelogregDf['predictTimeSecMean'], barWidth,
            yerr=openfacelogregDf['predictTimeSecStd'],
            label='OpenFace LogisticRegression',
            color=colors[4], ecolor='0.3', alpha=alpha)
    plt.bar(indices + 5 * barWidth, facenetsvmDf['predictTimeSecMean'], barWidth,
            yerr=facenetsvmDf['predictTimeSecStd'],
            label='Facenet SVM',
            color=colors[5], ecolor='0.3', alpha=alpha)
    plt.bar(indices + 6 * barWidth, facenetlinearsvmDf['predictTimeSecMean'], barWidth,
            yerr=facenetlinearsvmDf['predictTimeSecStd'],
            label='Facenet LinearSVC',
            color=colors[6], ecolor='0.3', alpha=alpha)
    plt.bar(indices + 7 * barWidth, facenetgaussiannbDf['predictTimeSecMean'], barWidth,
            yerr=facenetgaussiannbDf['predictTimeSecStd'],
            label='Facenet GaussianNB',
            color=colors[7], ecolor='0.3', alpha=alpha)    
    plt.bar(indices + 8 * barWidth, facenetdectreeDf['predictTimeSecMean'], barWidth,
            yerr=facenetdectreeDf['predictTimeSecStd'],
            label='Facenet DecisionTree',
            color=colors[8], ecolor='0.3', alpha=alpha)    
    plt.bar(indices + 9 * barWidth, facenetlogregDf['predictTimeSecMean'], barWidth,
            yerr=facenetlogregDf['predictTimeSecStd'],
            label='Facenet LogisticRegression',
            color=colors[9], ecolor='0.3', alpha=alpha) 

    box = ax.get_position()
    if largeFont:
        ax.set_position([box.x0, box.y0 + 0.11, box.width, box.height * 0.7])
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.45), ncol=8,
                   fancybox=True, shadow=True, fontsize=16)
    else:
        ax.set_position([box.x0, box.y0 + 0.05, box.width, box.height * 0.77])
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.37), ncol=8,
                   fancybox=True, shadow=True)
    plt.ylabel("Prediction Time (s)")
    plt.xlabel("Number of People")
    ax.set_xticks(indices + 4 * barWidth)
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
        f.write('OpenFace GaussianNB: {1}, {0}\n'.format(openfacegaussiannbDf['predictTimeSecStd'], openfacegaussiannbDf['predictTimeSecMean']))
        f.write('OpenFace DecisionTree: {1}, {0}\n'.format(openfacedectreeDf['predictTimeSecStd'], openfacedectreeDf['predictTimeSecMean']))
        f.write('OpenFace LogisticRegression: {1}, {0}\n'.format(openfacelogregDf['predictTimeSecStd'], openfacelogregDf['predictTimeSecMean']))
        f.write('Facenet SVM: {1}, {0}\n'.format(facenetsvmDf['predictTimeSecStd'], facenetsvmDf['predictTimeSecMean']))
        f.write('Facenet LinearSVC {1}, {0}\n'.format(facenetlinearsvmDf['predictTimeSecStd'], facenetlinearsvmDf['predictTimeSecMean']))
        f.write('Facenet GaussianNB {1}, {0}\n'.format(facenetgaussiannbDf['predictTimeSecStd'], facenetgaussiannbDf['predictTimeSecMean']))
        f.write('Facenet DecisionTree {1}, {0}\n'.format(facenetdectreeDf['predictTimeSecStd'], facenetdectreeDf['predictTimeSecMean']))
        f.write('Facenet LogisticRegression {1}, {0}\n'.format(facenetlogregDf['predictTimeSecStd'], facenetlogregDf['predictTimeSecMean']))


if __name__ == '__main__':
    main()
