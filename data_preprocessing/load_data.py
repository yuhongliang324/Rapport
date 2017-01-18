__author__ = 'yuhongliang324'
import numpy
import theano
import os
import sys
sys.path.append('../')
from data_preprocessing.utils import load_feature
from data_path import sample_10_root
from sklearn.preprocessing import normalize
import random

n_class = 7


def load(dirname, feature_name='hog', side='lr', min_step=76, norm=True):

    dyad_features = {}
    dyad_ratings = {}

    files = os.listdir(dirname)
    files.sort()
    for fn in files:
        print fn
        session_dir = os.path.join(dirname, fn)
        if not (os.path.isdir(session_dir) and fn.startswith('D')):
            continue
        dyad = int(fn[1:].split('S')[0])
        if 'D9S2' not in session_dir:
            continue
        features, ratings = load_dyad(session_dir, feature_name=feature_name, side=side, min_step=min_step, norm=norm)
        if min_step is not None:
            features = features[:, :min_step, :]
        if dyad in dyad_features:
            dyad_features[dyad] = numpy.concatenate((dyad_features[dyad], features), axis=0)
            dyad_ratings[dyad] = numpy.concatenate((dyad_ratings[dyad], ratings), axis=0)
        else:
            dyad_features[dyad] = features
            dyad_ratings[dyad] = ratings
    return dyad_features, dyad_ratings


def load_dyad(dirname, feature_name='hog', side='b', min_step=76, norm=True):
    def add_to_features(feat, rating, features, ratings, prev_step, min_step=76):
        if feat.shape[0] < min_step:
            return
        if prev_step is None:
            prev_step = feat.shape[0]
        elif feat.shape[0] != prev_step:
            return
        features.append(feat)
        ratings.append(rating)
        return prev_step
    features, ratings = [], []

    files = os.listdir(dirname)
    files.sort()
    prev_step = None
    for mat_name in files:
        if not mat_name.endswith('mat'):
            continue
        mat_file = os.path.join(dirname, mat_name)
        ret = load_feature(mat_file, feature_name=feature_name, side=side, only_suc=False)
        if ret is None:
            continue
        feat, _, rating = ret
        if side == 'lr':
            lfeat, rfeat = feat
            if norm:
                lfeat = normalize(lfeat)
                rfeat = normalize(rfeat)
            prev_step = add_to_features(lfeat, rating, features, ratings, prev_step, min_step=min_step)
            prev_step = add_to_features(rfeat, rating, features, ratings, prev_step, min_step=min_step)
        else:
            if norm:
                feat = normalize(feat)
            prev_step = add_to_features(feat, rating, features, ratings, prev_step, min_step=min_step)
    for f in features:
        print f.shape
    features = numpy.stack(features[:-1], axis=0).astype(theano.config.floatX)
    ratings = numpy.asarray(ratings[:-1], dtype=theano.config.floatX)

    return features, ratings


def load_pairs(dirname, feature_name='hog', side='lr', min_step=76, norm=True, n_class=1):
    dyad_X1 = {}
    dyad_X2 = {}
    dyad_gaps = {}

    files = os.listdir(dirname)
    files.sort()
    for fn in files:
        print fn
        session_dir = os.path.join(dirname, fn)
        if not (os.path.isdir(session_dir) and fn.startswith('D')):
            continue
        dyad = int(fn[1:].split('S')[0])
        X1, X2, y = load_dyad_pairs(session_dir, feature_name=feature_name, side=side, min_step=min_step,
                                    norm=norm, n_class=n_class)
        if min_step is not None:
            X1 = X1[:, :min_step, :]
            X2 = X2[:, :min_step, :]
        if dyad in dyad_X1:
            dyad_X1[dyad] = numpy.concatenate((dyad_X1[dyad], X1), axis=0)
            dyad_X2[dyad] = numpy.concatenate((dyad_X2[dyad], X2), axis=0)
            dyad_gaps[dyad] = numpy.concatenate((dyad_gaps[dyad], y), axis=0)
        else:
            dyad_X1[dyad] = X1
            dyad_X2[dyad] = X2
            dyad_gaps[dyad] = y
    return dyad_X1, dyad_X2, dyad_gaps


def load_dyad_pairs(dirname, feature_name='hog', side='b', min_step=76, norm=True, prob=.4, n_class=1):
    features1, features2, gaps = [], [], []

    def add_to_pair(features, ratings):
        def comp(a, b, threshold=0.5):
            if n_class == 1:
                return a - b
            elif n_class == 2:
                if a < b:
                    return 0
                else:
                    return 1
            else:
                if abs(a - b) < threshold:
                    return 2
                elif a < b:
                    return 0
                else:
                    return 1

        for i in xrange(ratings.shape[0] - 1):
            for j in xrange(i + 1, ratings.shape[0]):
                if abs(ratings[i] - ratings[j]) < 3:
                    continue
                r = random.random()
                if r > prob:
                    continue
                r = random.random()
                if r < 0.5:
                    f1, f2 = features[i], features[j]
                    gap = comp(ratings[i], ratings[j])

                else:
                    f1, f2 = features[j], features[i]
                    gap = comp(ratings[j], ratings[i])
                features1.append(f1)
                features2.append(f2)
                gaps.append(gap)

    if side == 'lr':
        features, ratings = load_dyad(dirname, feature_name, 'l', min_step=min_step, norm=norm)
        add_to_pair(features, ratings)
        features, ratings = load_dyad(dirname, feature_name, 'r', min_step=min_step, norm=norm)
        add_to_pair(features, ratings)
    else:
        features, ratings = load_dyad(dirname, feature_name, side=side, min_step=min_step, norm=norm)
        add_to_pair(features, ratings)
    X1 = numpy.stack(features1, axis=0).astype(theano.config.floatX)
    X2 = numpy.stack(features2, axis=0).astype(theano.config.floatX)
    y = numpy.asarray(gaps, dtype=theano.config.floatX)
    return X1, X2, y


def test1():
    load(sample_10_root)

