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

nPplVals = [10, 50, 100, 200, 400]
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

    eigenFacesDf = None
    fishFacesDf = None
    lbphFacesDf = None
    openfaceCPUsvmDf = None
    openfaceCPUlinearsvmDf = None
    facenetlinearsvmDf = None

    print("Getting lfwPpl")
    lfwPplCache = os.path.join(args.workDir, 'lfwPpl.pkl')
    lfwPpl = cacheToFile(lfwPplCache)(getLfwPplSorted)(args.lfwDlibAligned)

    print("Getting lfwMtcnnPpl")
    lfwMtcnnPplCache = os.path.join(args.workDir, 'lfwMtcnnPpl.pkl')
    lfwMtcnnPpl = cacheToFile(lfwMtcnnPplCache)(getLfwPplSorted)(args.lfwMtcnnAligned)

    def eigenfaces():
        print("Eigenfaces Experiment")
        global eigenFacesDf
        cls = cv2.createEigenFaceRecognizer()
        cache = os.path.join(args.workDir, 'eigenFacesExp.pkl')
        eigenFacesDf = cacheToFile(cache)(opencvExp)(lfwPpl, cls)

    def fisherfaces():
        print("Fisherfaces Experiment")
        global fishFacesDf
        cls = cv2.createFisherFaceRecognizer()
        cache = os.path.join(args.workDir, 'fisherFacesExp.pkl')
        fishFacesDf = cacheToFile(cache)(opencvExp)(lfwPpl, cls)

    def lbph():
        print("LBPH Experiment")
        global lbphFacesDf
        cls = cv2.createLBPHFaceRecognizer()
        cache = os.path.join(args.workDir, 'lbphExp.pkl')
        lbphFacesDf = cacheToFile(cache)(opencvExp)(lfwPpl, cls)

    def openfacesvm():
        print("OpenFace SVM Experiment")
        global openfaceCPUsvmDf
        net = openface.TorchNeuralNet(openfaceModelPath, 96, cuda=False)
        cls = SVC(C=1, kernel='linear')
        cache = os.path.join(args.workDir, 'openface.cpu.svm.pkl')
        openfaceCPUsvmDf = cacheToFile(cache)(openfaceExp)(lfwPpl, net, cls)

    def openfacelinsvm():
        print("OpenFaceLinearSVC Experiment")
        global openfaceCPUlinearsvmDf
        net = openface.TorchNeuralNet(openfaceModelPath, 96, cuda=False)
        cls = LinearSVC(C=1, multi_class='ovr')
        cache = os.path.join(args.workDir, 'openface.cpu.linearsvm.pkl')
        openfaceCPUlinearsvmDf = cacheToFile(cache)(openfaceExp)(lfwPpl, net, cls)

    def facenetlinsvm():
        print("Facenet LinearSVC Experiment")
        global facenetlinearsvmDf
        cls = LinearSVC(C=1, multi_class='ovr')
        cache = os.path.join(args.workDir, 'facenet.linearsvm.pkl')
        facenetlinearsvmDf = cacheToFile(cache)(facenetExp)(lfwMtcnnPpl, facenetModelDir, cls)

    threads = []

    t1 = Thread(target=eigenfaces, args=())
    t2 = Thread(target=fisherfaces, args=())
    t3 = Thread(target=lbph, args=())
    t4 = Thread(target=openfacesvm, args=())
    t5 = Thread(target=openfacelinsvm, args=())
    t6 = Thread(target=facenetlinsvm, args=())

    threads.append(t1)
    threads.append(t2)    
    threads.append(t3)
    threads.append(t4)
    threads.append(t5)
    threads.append(t6)

    # Start all threads
    for x in threads:
        x.start()

    # Wait for all of them to finish
    for x in threads:
        x.join()


    plotAccuracy(args.workDir, args.largeFont,
                    eigenFacesDf, fishFacesDf, lbphFacesDf,
                    openfaceCPUsvmDf, openfaceCPUlinearsvmDf, facenetlinearsvmDf)
    plotTrainingTime(args.workDir, args.largeFont,
                        eigenFacesDf, fishFacesDf, lbphFacesDf,
                        openfaceCPUsvmDf, openfaceCPUlinearsvmDf, facenetlinearsvmDf)
    plotPredictionTime(args.workDir, args.largeFont,
                        eigenFacesDf, fishFacesDf, lbphFacesDf,
                        openfaceCPUsvmDf, openfaceCPUlinearsvmDf, facenetlinearsvmDf)

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


def plotAccuracy(workDir, largeFont, eigenFacesDf, fishFacesDf, lbphFacesDf,
                 openfaceCPUsvmDf, openfaceCPUlinearsvmDf, facenetlinearsvmDf):
    indices = eigenFacesDf.index.values
    barWidth = 0.15

    if largeFont:
        fig = plt.figure(figsize=(10, 5))
    else:
        fig = plt.figure(figsize=(14, 4))
    ax = fig.add_subplot(111)
    plt.bar(indices, eigenFacesDf['accsMean'], barWidth,
            yerr=eigenFacesDf['accsStd'], label='Eigenfaces',
            color=colors[0], ecolor='0.3', alpha=alpha)
    plt.bar(indices + barWidth, fishFacesDf['accsMean'], barWidth,
            yerr=fishFacesDf['accsStd'], label='Fisherfaces',
            color=colors[1], ecolor='0.3', alpha=alpha)
    plt.bar(indices + 2 * barWidth, lbphFacesDf['accsMean'], barWidth,
            yerr=lbphFacesDf['accsStd'], label='LBPH',
            color=colors[2], ecolor='0.3', alpha=alpha)
    plt.bar(indices + 3 * barWidth, openfaceCPUsvmDf['accsMean'], barWidth,
            yerr=openfaceCPUsvmDf['accsStd'], label='OpenFace',
            color=colors[3], ecolor='0.3', alpha=alpha)
    plt.bar(indices + 4 * barWidth, openfaceCPUlinearsvmDf['accsMean'], barWidth,
            yerr=openfaceCPUlinearsvmDf['accsStd'], label='OpenFace LinearSVC',
            color=colors[4], ecolor='0.3', alpha=alpha)
    plt.bar(indices + 5 * barWidth, facenetlinearsvmDf['accsMean'], barWidth,
            yerr=facenetlinearsvmDf['accsStd'], label='Facenet LinearSVC',
            color=colors[5], ecolor='0.3', alpha=alpha)

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
        f.write('Eigenfaces: {1}, {0}\n'.format(eigenFacesDf['accsStd'], eigenFacesDf['accsMean']))
        f.write('Fisherfaces: {1}, {0}\n'.format(fishFacesDf['accsStd'], fishFacesDf['accsMean']))
        f.write('LBPH: {1}, {0}\n'.format(lbphFacesDf['accsStd'], lbphFacesDf['accsMean']))
        f.write('OpenFace SVM: {1}, {0}\n'.format(openfaceCPUsvmDf['accsStd'], openfaceCPUsvmDf['accsMean']))
        f.write('OpenFace LinearSVC: {1}, {0}\n'.format(openfaceCPUlinearsvmDf['accsStd'], openfaceCPUlinearsvmDf['accsMean']))
        f.write('Facenet LinearSVC {1}, {0}\n'.format(facenetlinearsvmDf['accsStd'], facenetlinearsvmDf['accsMean']))


def plotTrainingTime(workDir, largeFont, eigenFacesDf, fishFacesDf, lbphFacesDf,
                     openfaceCPUsvmDf, openfaceCPUlinearsvmDf, facenetlinearsvmDf):
    indices = eigenFacesDf.index.values
    barWidth = 0.15

    fig = plt.figure(figsize=(14, 4))
    ax = fig.add_subplot(111)
    plt.bar(indices, eigenFacesDf['trainTimeSecMean'], barWidth,
            yerr=eigenFacesDf['trainTimeSecStd'], label='Eigenfaces',
            color=colors[0], ecolor='0.3', alpha=alpha)
    plt.bar(indices + barWidth, fishFacesDf['trainTimeSecMean'], barWidth,
            yerr=fishFacesDf['trainTimeSecStd'], label='Fisherfaces',
            color=colors[1], ecolor='0.3', alpha=alpha)
    plt.bar(indices + 2 * barWidth, lbphFacesDf['trainTimeSecMean'], barWidth,
            yerr=lbphFacesDf['trainTimeSecStd'], label='LBPH',
            color=colors[2], ecolor='0.3', alpha=alpha)
    plt.bar(indices + 3 * barWidth, openfaceCPUsvmDf['trainTimeSecMean'], barWidth,
            yerr=openfaceCPUsvmDf['trainTimeSecStd'],
            label='OpenFace',
            color=colors[3], ecolor='0.3', alpha=alpha)
    plt.bar(indices + 4 * barWidth, openfaceCPUlinearsvmDf['trainTimeSecMean'], barWidth,
            yerr=openfaceCPUlinearsvmDf['trainTimeSecStd'],
            label='OpenFace LinearSVC',
            color=colors[4], ecolor='0.3', alpha=alpha)
    plt.bar(indices + 5 * barWidth, facenetlinearsvmDf['trainTimeSecMean'], barWidth,
            yerr=facenetlinearsvmDf['trainTimeSecStd'],
            label='Facenet LinearSVC',
            color=colors[5], ecolor='0.3', alpha=alpha)


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
        f.write('Eigenfaces: {1}, {0}\n'.format(eigenFacesDf['trainTimeSecStd'], eigenFacesDf['trainTimeSecMean']))
        f.write('Fisherfaces: {1}, {0}\n'.format(fishFacesDf['trainTimeSecStd'], fishFacesDf['trainTimeSecMean']))
        f.write('LBPH: {1}, {0}\n'.format(lbphFacesDf['trainTimeSecStd'], lbphFacesDf['trainTimeSecMean']))
        f.write('OpenFace SVM: {1}, {0}\n'.format(openfaceCPUsvmDf['trainTimeSecStd'], openfaceCPUsvmDf['trainTimeSecMean']))
        f.write('OpenFace LinearSVC: {1}, {0}\n'.format(openfaceCPUlinearsvmDf['trainTimeSecStd'], openfaceCPUlinearsvmDf['trainTimeSecMean']))
        f.write('Facenet LinearSVC: {1}, {0}\n'.format(facenetlinearsvmDf['trainTimeSecStd'], facenetlinearsvmDf['trainTimeSecMean']))


def plotPredictionTime(workDir, largeFont, eigenFacesDf, fishFacesDf, lbphFacesDf,
                       openfaceCPUsvmDf, openfaceCPUlinearsvmDf, facenetlinearsvmDf):
    indices = eigenFacesDf.index.values
    barWidth = 0.15

    fig = plt.figure(figsize=(14, 4))
    ax = fig.add_subplot(111)
    plt.bar(indices, eigenFacesDf['predictTimeSecMean'], barWidth,
            yerr=eigenFacesDf['predictTimeSecStd'], label='Eigenfaces',
            color=colors[0], ecolor='0.3', alpha=alpha)
    plt.bar(indices + barWidth, fishFacesDf['predictTimeSecMean'], barWidth,
            yerr=fishFacesDf['predictTimeSecStd'], label='Fisherfaces',
            color=colors[1], ecolor='0.3', alpha=alpha)
    plt.bar(indices + 2 * barWidth, lbphFacesDf['predictTimeSecMean'], barWidth,
            yerr=lbphFacesDf['predictTimeSecStd'], label='LBPH',
            color=colors[2], ecolor='0.3', alpha=alpha)
    plt.bar(indices + 3 * barWidth, openfaceCPUsvmDf['predictTimeSecMean'], barWidth,
            yerr=openfaceCPUsvmDf['predictTimeSecStd'],
            label='OpenFace',
            color=colors[3], ecolor='0.3', alpha=alpha)
    plt.bar(indices + 4 * barWidth, openfaceCPUlinearsvmDf['predictTimeSecMean'], barWidth,
            yerr=openfaceCPUlinearsvmDf['predictTimeSecStd'],
            label='OpenFace LinearSVC',
            color=colors[4], ecolor='0.3', alpha=alpha)
    plt.bar(indices + 5 * barWidth, facenetlinearsvmDf['predictTimeSecMean'], barWidth,
            yerr=facenetlinearsvmDf['predictTimeSecStd'],
            label='Facenet LinearSVC',
            color=colors[5], ecolor='0.3', alpha=alpha)


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
        f.write('Eigenfaces: {1}, {0}\n'.format(eigenFacesDf['predictTimeSecStd'], eigenFacesDf['predictTimeSecMean']))
        f.write('Fisherfaces: {1}, {0}\n'.format(fishFacesDf['predictTimeSecStd'], fishFacesDf['predictTimeSecMean']))
        f.write('LBPH: {1}, {0}\n'.format(lbphFacesDf['predictTimeSecStd'], lbphFacesDf['predictTimeSecMean']))
        f.write('OpenFace SVM: {1}, {0}\n'.format(openfaceCPUsvmDf['predictTimeSecStd'], openfaceCPUsvmDf['predictTimeSecMean']))
        f.write('OpenFace LinearSVC: {1}, {0}\n'.format(openfaceCPUlinearsvmDf['predictTimeSecStd'], openfaceCPUlinearsvmDf['predictTimeSecMean']))
        f.write('Facenet LinearSVC: {1}, {0}\n'.format(facenetlinearsvmDf['predictTimeSecStd'], facenetlinearsvmDf['predictTimeSecMean']))


if __name__ == '__main__':
    main()
