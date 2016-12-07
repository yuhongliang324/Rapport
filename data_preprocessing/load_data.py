__author__ = 'yuhongliang324'
import numpy
import theano
import os
import sys
sys.path.append('../')
from data_preprocessing.utils import load_feature
from data_path import sample_10_root

n_class = 7


def load(dirname, feature_name='hog', side='l', min_step=76):

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
        features, ratings = load_dyad(session_dir, feature_name=feature_name, side=side)
        if min_step is not None:
            features = features[:, :76, :]
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
    prev_step = None
    for mat_name in files:
        if not mat_name.endswith('mat'):
            continue
        mat_file = os.path.join(dirname, mat_name)
        feat, _, rating = load_feature(mat_file, feature_name=feature_name, side=side, only_suc=False)
        if feat.shape[0] == 0:
            continue
        if prev_step is None:
            prev_step = feat.shape[0]
        elif feat.shape[0] != prev_step:
            continue
        features.append(feat)
        ratings.append(rating)

    features = numpy.stack(features[:-1], axis=0).astype(theano.config.floatX)
    ratings = numpy.asarray(ratings[:-1], dtype=theano.config.floatX)

    return features, ratings


def test1():
    load(sample_10_root)
