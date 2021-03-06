__author__ = 'yuhongliang324'
import sys
sys.path.append('..')
import os
from scipy.io import loadmat
import numpy
import theano
from data_path import audio_root, ratings_file
from sklearn.preprocessing import normalize


names = ['F0', 'VUV', 'NAQ', 'QOQ', 'H1H2', 'PSP', 'MDQ', 'peakSlope',
         'Rd', 'Rd_conf', 'creak', 'MCEP_0', 'MCEP_1', 'MCEP_2', 'MCEP_3', 'MCEP_4', 'MCEP_5',
         'MCEP_6', 'MCEP_7', 'MCEP_8', 'MCEP_9', 'MCEP_10', 'MCEP_11', 'MCEP_12', 'MCEP_13', 'MCEP_14',
         'MCEP_15', 'MCEP_16', 'MCEP_17', 'MCEP_18', 'MCEP_19', 'MCEP_20', 'MCEP_21', 'MCEP_22', 'MCEP_23',
         'MCEP_24', 'HMPDM_0', 'HMPDM_1', 'HMPDM_2', 'HMPDM_3', 'HMPDM_4', 'HMPDM_5', 'HMPDM_6',
         'HMPDM_7', 'HMPDM_8', 'HMPDM_9', 'HMPDM_10', 'HMPDM_11', 'HMPDM_12', 'HMPDM_13', 'HMPDM_14',
         'HMPDM_15', 'HMPDM_16',  'HMPDM_17', 'HMPDM_18', 'HMPDM_19', 'HMPDM_20', 'HMPDM_21',
         'HMPDM_22', 'HMPDM_23', 'HMPDM_24', 'HMPDD_0', 'HMPDD_1', 'HMPDD_2', 'HMPDD_3', 'HMPDD_4',
         'HMPDD_5', 'HMPDD_6', 'HMPDD_7', 'HMPDD_8', 'HMPDD_9', 'HMPDD_10', 'HMPDD_11', 'HMPDD_12']


def load_audio_features(root=audio_root):
    slices = []
    features = []

    files = os.listdir(root)
    files.sort()
    for dname in files:
        dpath = os.path.join(root, dname)
        if not os.path.isdir(dpath):
            continue
        print dname
        files1 = os.listdir(dpath)
        files1.sort()
        for mat_name in files1:
            if not mat_name.endswith('mat'):
                continue
            sp = mat_name.split('_')
            dyad = int(sp[0][1:])
            session = int(sp[1][1:])
            slice = int(sp[2])
            slices.append((dyad, session, slice))
            mat_path = os.path.join(dpath, mat_name)
            data = loadmat(mat_path)
            feat = data['features']
            feat = numpy.percentile(feat, 90, axis=0)
            features.append(feat)
    X = numpy.stack(features, axis=0).astype(theano.config.floatX)
    X[numpy.isneginf(X)] = -1.
    return X, slices


def load_ratings(ratings_file=ratings_file):
    reader = open(ratings_file)
    lines = reader.readlines()
    reader.close()
    lines = map(lambda x: x.strip(), lines)
    slice_rating = {}
    for line in lines:
        sp = line.split(',')
        dyad, session, slice = int(sp[0]), int(sp[1]), int(sp[2])
        rating = float(sp[3])
        slice_rating[(dyad, session, slice)] = rating
    return slice_rating


def get_PCC():
    X, slices = load_audio_features()
    slice_rating = load_ratings()
    y = []
    for slice in slices:
        if not slice in slice_rating:
            rating = 3.5
        else:
            rating = slice_rating[slice]
        y.append(rating)
    y = numpy.asarray(y, dtype=theano.config.floatX)

    def PCC(X, y, topK=20):
        X_bar = normalize(X - numpy.mean(X, axis=0), axis=0)
        y_bar = normalize(y - numpy.mean(y))
        y_bar = numpy.squeeze(y_bar)
        PCC = numpy.dot(X_bar.T, y_bar)
        PCC_abs = abs(PCC)
        PCC_sign = numpy.sign(PCC)
        ind = numpy.argsort(PCC_abs)[::-1]
        print '\t',
        for j in xrange(topK):
            print '%s %.3f;' % (names[ind[j]], PCC_abs[ind[j]] * PCC_sign[ind[j]]),
        print
    PCC(X, y)


def test1():
    get_PCC()


def test2():
    load_audio_features()


if __name__ == '__main__':
    test1()
