__author__ = 'yuhongliang324'
import numpy
import theano
import os
import sys
sys.path.append('../')
from data_preprocessing.utils import load_feature
from data_path import sample_10_root

n_class = 7


def load(dirname, feature_name='hog', side='b'):

    dyad_features = {}
    dyad_ratings = {}

    files = os.listdir(dirname)
    files.sort()
    for fn in files:
        print fn
        if not (os.path.isdir(fn) and fn.startswith('D')):
            continue
        dyad = int(fn[1:].split('S')[0])
        features, ratings = load_dyad(os.path.join(dirname, fn), feature_name=feature_name, side=side)
        if dyad in dyad_features:
            dyad_features[dyad] = numpy.concatenate((dyad_features[dyad], features), axis=0)
            dyad_ratings[dyad] = numpy.concatenate((dyad_ratings[dyad], ratings), axis=0)
        else:
            dyad_features[dyad] = features
            dyad_ratings[dyad] = ratings
    return dyad_features, dyad_ratings


def load_dyad(dirname, feature_name='hog', side='b'):
    features, ratings = [], []

    files = os.listdir(dirname)
    files.sort()
    for mat_name in files:
        if not mat_name.endswith('mat'):
            continue
        mat_file = os.path.join(dirname, mat_name)
        feat, _, rating = load_feature(mat_file, feature_name=feature_name, side=side, only_suc=False)
        if feat.shape[0] == 0:
            continue
        features.append(feat)
        ratings.append(rating)

    features = numpy.asarray(features, dtype=theano.config.floatX)
    ratings = numpy.asarray(ratings, dtype=theano.config.floatX)

    return features, ratings


def test1():
    load(sample_10_root)

