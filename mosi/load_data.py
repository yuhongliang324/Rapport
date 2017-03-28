__author__ = 'yuhongliang324'

from preprocess import data_root, split_file
import cPickle as pickle
import os
import numpy


def load(feature_name='audio', category=False):
    files = os.listdir(data_root)
    files.sort()
    session_Xs = {}
    session_y = {}
    for fn in files:
        if not fn.endswith('pkl'):
            continue
        print fn
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
            X[numpy.isnan(X)] = 0.
            X[numpy.isneginf(X)] = -1.
            session_Xs[videoID].append(X)
            if not category:
                session_y[videoID].append(cont['label'])
            else:
                if cont['label'] > 0:
                    l = 1
                else:
                    l = 0
                session_y[videoID].append(l)
    return session_Xs, session_y


def load_split():
    reader = open(split_file)
    lines = reader.readlines()
    reader.close()
    lines = map(lambda x: x.strip(), lines)
    tests = lines[0].split()
    trains = lines[1].split()
    return tests, trains