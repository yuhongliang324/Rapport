__author__ = 'yuhongliang324'
import os
import cPickle as pickle
from scipy.io import loadmat
import numpy


raw_data_root = '/multicomp/datasets/mosi'
raw_openface_pkl = os.path.join(raw_data_root, 'OpenFaceFeatures.pkl')
raw_audio_root = os.path.join(raw_data_root, 'Audio/segment')
raw_facet_root = os.path.join(raw_data_root, 'FACET_GIOTA')

data_root = '/multicomp/datasets/mosi/Features'


def get_openface_features():
    reader = open(raw_openface_pkl)
    data = pickle.load(reader)
    reader.close()
    print 'OpenFace features loaded'

    # key of data: video ID
    # key of data[videoID]: segment ID
    # key of data[videoID][segID]: 'start_time', 'openface', 'end_time', 'sentiment'
    return data


def get_audio_features():
    data = {}
    mats = os.listdir(raw_audio_root)
    mats.sort()
    for mat in mats:
        if not mat.endswith('mat'):
            continue
        mat_path = os.path.join(raw_audio_root, mat)
        d = loadmat(mat_path)
        feat = d['features']
        videoID = mat[:11]
        segID = mat[12:-4]
        if videoID not in data:
            data[videoID] = {}
        data[videoID][segID] = feat
    return data


# video_range: the output of get_openface_features
def get_facet_features(video_range):
    files = os.listdir(raw_facet_root)
    files.sort()
    data = {}
    for csv in files:
        if not csv.endswith('csv'):
            continue
        videoID = csv[:11]
        data[videoID] = {}
        video_path = os.path.join(raw_facet_root, csv)
        features = numpy.loadtxt(video_path, skiprows=1, delimiter=',')
        segID_cont = video_range[videoID]
        for segID, cont in segID_cont.items():
            start_time, end_time = cont['start_time'], cont['end_time']
            start_frame, end_frame = int(start_time * 30), int(end_time * 30)
            feat = features[start_frame: end_frame + 1]
            data[videoID][segID] = feat
    return data


def write_features(data_openface, data_audio, data_facet):
    videoIDs = data_openface.keys()
    for videoID in videoIDs:
        print videoID
        video_path = os.path.join(data_root, videoID + '.pkl')
        writer = open(video_path, 'w')
        segID_cont = data_openface[videoID]
        data = {}
        for segID, cont in segID_cont.items():
            new_cont = {'start_time': cont['start_time'], 'end_time': cont['end_time'], 'label': cont['sentiment']}
            feat_openface = cont['openface']
            new_cont['openface'] = numpy.asarray(feat_openface, dtype=numpy.float32)
            new_cont['facet'] = numpy.asarray(data_facet[videoID][segID], dtype=numpy.float32)
            new_cont['audio'] = numpy.asarray(data_audio[videoID][segID], dtype=numpy.float32)
            data[segID] = new_cont
        pickle.dump(data, writer)
        writer.close()
