__author__ = 'yuhongliang324'

from preprocess import data_root
import cPickle as pickle
import os


def load(feat_name='audio', category=False):
    files = os.listdir(data_root)
    files.sort()
    session_Xs = {}
    session_y = {}
    for fn in files:
        if not fn.endswith('pkl'):
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
            session_Xs[videoID].append(cont[feat_name])
            if not category:
                session_y[videoID].append(cont['label'])
            else:
                if cont['label'] > 0:
                    l = 1
                else:
                    l = 0
                session_y[videoID].append(l)
    return session_Xs, session_y
