__author__ = 'yuhongliang324'

from preprocess import data_root, data_root_hog, split_file
import cPickle as pickle
import os
import numpy
import sys
import cPickle
from preprocess import dict_pkl
sys.path.append('..')
from utils import standardize
import theano


def load(feature_name='audio', category=False):
    if feature_name == 'hog':
        files = os.listdir(data_root_hog)
    else:
        files = os.listdir(data_root)
    files.sort()
    session_Xs = {}
    session_y = {}
    for fn in files:
        if not fn.endswith('pkl'):
            continue
        print fn
        if fn.startswith('c5xsKMxpXnc'):
            continue
        videoID = fn[:-4]
        session_Xs[videoID] = []
        session_y[videoID] = []
        pkl_path = os.path.join(data_root, fn)
        reader = open(pkl_path)
        segID_cont = pickle.load(reader)
        reader.close()
        conts = segID_cont.values()
        for cont in conts:
            X = cont[feature_name]
            if len(X.shape) == 1:
                continue
            X[numpy.isnan(X)] = 0.
            X[numpy.isneginf(X)] = 0.
            X[numpy.isinf(X)] = 0.
            session_Xs[videoID].append(X)
            if not category:
                session_y[videoID].append(cont['label'])
            else:
                if cont['label'] > 0:
                    l = 1
                else:
                    l = 0
                session_y[videoID].append(l)
        session_Xs[videoID] = standardize(session_Xs[videoID])
    return session_Xs, session_y


def load_split():
    reader = open(split_file)
    lines = reader.readlines()
    reader.close()
    lines = map(lambda x: x.strip(), lines)
    tests = lines[0].split()
    trains = lines[1].split()
    return tests, trains


def load_dict(vec_file=dict_pkl):
    reader = open(vec_file, 'rb')
    token_ID, E = cPickle.load(reader)
    reader.close()
    return token_ID, E.astype(theano.config.floatX)
