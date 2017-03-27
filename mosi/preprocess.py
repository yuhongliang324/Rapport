__author__ = 'yuhongliang324'
import os
import cPickle as pickle
import numpy


raw_data_root = '/multicomp/datasets/mosi'
raw_openface_pkl = os.path.join(raw_data_root, 'OpenFaceFeatures.pkl')
raw_audio_root = os.path.join(raw_data_root, 'Audio/segment')


def get_openface_features():
    reader = open(raw_openface_pkl)
    data = pickle.load(reader)
    reader.close()

    # key of data: video ID
    # key of data[videoID]: segment ID
    # key of data[videoID][segID]: 'start_time', 'openface', 'end_time', 'sentiment'
    return data


def get_audio_features():
    mats = os.listdir(raw_audio_root)
    mats.sort()
    for mat in mats:
        if not mat.endswith('mat'):
            continue
        if not mat[11] == '_':
            print mat[11]


get_audio_features()
