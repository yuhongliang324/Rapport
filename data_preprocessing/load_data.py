__author__ = 'yuhongliang324'
import os
import sys

import numpy
import theano

sys.path.append('../')

from utils import get_ratings, interpolate_features
from data_path import sample_10_root, audio_root, ratings_file, ratings_best3_file, rating_class_file, rating_class_best3_file
from sklearn.preprocessing import normalize
from scipy.io import loadmat

n_class = 5


def get_valid_slices():
    slice_ratings = get_ratings()
    candidates = slice_ratings.keys()
    valid_slices = set()
    for cand in candidates:
        sp = cand.split('_')
        dyad, session, slice = sp[0], sp[1], sp[2].zfill(3)
        mat_root = os.path.join(audio_root, 'D' + dyad + 'S' + session)
        lmat_path = os.path.join(mat_root, 'D' + dyad + '_S' + session + '_' + slice + '_left.mat')
        rmat_path = os.path.join(mat_root, 'D' + dyad + '_S' + session + '_' + slice + '_right.mat')
        if os.path.isfile(lmat_path) and os.path.isfile(rmat_path):
            valid_slices.add(cand)
    return valid_slices


def load_vision(dirname, feature_name='hog', side='ba', min_step=76, normalization=True, best3=False, category=False):
    valid_slices = get_valid_slices()
    slice_ratings = load_ratings(category=category, best3=best3)

    dyad_features = {}
    dyad_ratings = {}
    dyad_slices = {}

    files = os.listdir(dirname)
    files.sort()
    for fn in files:
        print fn
        session_dir = os.path.join(dirname, fn)
        if not (os.path.isdir(session_dir) and fn.startswith('D')):
            continue
        dyad = int(fn[1:].split('S')[0])
        features, ratings, slices = load_dyad_vision(session_dir, slice_ratings, feature_name=feature_name, side=side,
                                                     min_step=min_step, normalization=normalization,
                                                     valid_slices=valid_slices)
        if features is None:
            continue
        if min_step is not None:
            features = features[:, :min_step, :]
        if dyad in dyad_features:
            dyad_features[dyad] = numpy.concatenate((dyad_features[dyad], features), axis=0)
            dyad_ratings[dyad] = numpy.concatenate((dyad_ratings[dyad], ratings), axis=0)
            for slice in slices:
                dyad_slices[dyad].append(slice)
        else:
            dyad_features[dyad] = features
            dyad_ratings[dyad] = ratings
            dyad_slices[dyad] = slices
    return dyad_features, dyad_ratings, dyad_slices


def load_dyad_vision(dirname, slice_rating, feature_name='hog', side='ba', min_step=76, normalization=True, valid_slices=None):
    def add_to_features(feat, rating, slice, features, ratings, slices, prev_step, min_step=76):
        if feat.shape[0] < min_step:
            return prev_step
        if prev_step is None:
            prev_step = feat.shape[0]
        elif feat.shape[0] < prev_step:
            return prev_step
        start = (feat.shape[0] - prev_step) // 2
        feat = feat[start: start + min_step]
        features.append(feat)
        ratings.append(rating)
        slices.append(slice)
        return prev_step
    features, ratings = [], []
    slices = []

    files = os.listdir(dirname)
    files.sort()
    prev_step = None
    for mat_name in files:
        if not mat_name.endswith('mat'):
            continue
        tmp = mat_name[:-4]
        sp = tmp.split('_')
        dyad, session, slice_id = sp[0][1:], sp[1][1:], str(int(sp[2]))
        if valid_slices is not None and dyad + '_' + session + '_' + slice_id not in valid_slices:
            continue
        mat_file = os.path.join(dirname, mat_name)
        ret = load_feature_vision(mat_file, feature_name=feature_name, side=side)
        if ret is None:
            continue
        feat, _ = ret

        slice_tup = (dyad, session, slice_id)
        rating = slice_rating[(int(dyad), int(session), int(slice_id))]

        if side == 'lr':
            lfeat, rfeat = feat
            if normalize:
                if feature_name == 'gemo':
                    lfeat = normalize(lfeat, norm='l1', axis=0)
                    lfeat = normalize(lfeat)
                else:
                    lfeat = normalize(lfeat)
                if feature_name == 'gemo':
                    rfeat = normalize(rfeat, norm='l1', axis=0)
                    rfeat = normalize(rfeat)
                else:
                    rfeat = normalize(rfeat)
            prev_step = add_to_features(lfeat, rating, slice_tup, features, ratings, slices,
                                        prev_step, min_step=min_step)
            prev_step = add_to_features(rfeat, rating, slice_tup, features, ratings, slices,
                                        prev_step, min_step=min_step)
        else:
            if normalization:
                feat = normalize(feat)
            prev_step = add_to_features(feat, rating, slice_tup, features, ratings, slices,
                                        prev_step, min_step=min_step)
    if len(features) == 0:
        return None, None, None
    features = numpy.stack(features[:-1], axis=0).astype(theano.config.floatX)
    ratings = numpy.asarray(ratings[:-1], dtype=theano.config.floatX)

    return features, ratings, slices


# side: l - left only, r - right only, lr - left and right, b - concatenation of lr, ba - adding of lr
def load_feature_vision(mat_file, feature_name='hog', side='ba'):
    if feature_name == 'au' or feature_name == 'AU':
        fn = 'gemo'
    else:
        fn = feature_name
    lfeat_name = 'left_' + fn + '_feature'
    rfeat_name = 'right_' + fn + '_feature'
    data = loadmat(mat_file)
    lfeat, rfeat = data[lfeat_name], data[rfeat_name]
    if feature_name == 'au':
        lfeat, rfeat = lfeat[-35:], rfeat[-35:]
    lsuc, rsuc = numpy.squeeze(data['left_success']), numpy.squeeze(data['right_success'])
    lfeat = interpolate_features(lfeat, lsuc)
    rfeat = interpolate_features(rfeat, rsuc)
    if side == 'l':
        if lfeat is None or lfeat.dtype == numpy.int16:
            return None
        return lfeat, lsuc
    elif side == 'r':
        if rfeat is None or rfeat.dtype == numpy.int16:
            return None
        return rfeat, rsuc
    elif side == 'lr':
        if lfeat is None or rfeat is None or lfeat.dtype == numpy.int16 or rfeat.dtype == numpy.int16:
            return None
        return (lfeat, rfeat), (lsuc, rsuc)
    else:
        if lfeat is None or rfeat is None or lfeat.dtype == numpy.int16 or rfeat.dtype == numpy.int16:
            return None
        if 'a' in side:  # Use add for both sides
            feat = lfeat + rfeat
        else:
            feat = numpy.concatenate((lfeat, rfeat), axis=1)
        suc = numpy.logical_and(lsuc == 1, rsuc == 1)
        return feat, suc


def load_ratings(category=False, best3=False):
    if not category:
        if not best3:
            rpath = ratings_file
        else:
            rpath = ratings_best3_file
    else:
        if not best3:
            rpath = rating_class_file
        else:
            rpath = rating_class_best3_file
    reader = open(rpath)
    lines = reader.readlines()
    reader.close()
    lines = map(lambda x: x.strip(), lines)

    slice_rating = {}

    for line in lines:
        sp = line.split(',')
        dyad, session, slice = int(sp[0]), int(sp[1]), int(sp[2])
        if not category:
            rating = float(sp[3])
        else:
            rating = int(sp[3])
        slice_rating[(dyad, session, slice)] = rating
    return slice_rating


def load_audio(root=audio_root, side='b', num_frame=300, normalization=True, best3=False, category=False):
    dyad_features = {}
    dyad_slices = {}
    dyad_ratings = {}

    valid_slices = get_valid_slices()
    slice_rating = load_ratings(category=category, best3=best3)

    files = os.listdir(root)
    files.sort()
    for dname in files:
        dpath = os.path.join(root, dname)
        if not os.path.isdir(dpath):
            continue
        print dname
        dyad = int(dname[1:].split('S')[0])
        X, slices = load_dyad_audio(dpath, side=side, num_frame=num_frame, valid_slices=valid_slices,
                                    normalization=normalization)
        if X is None:
            continue
        ratings = []
        for slice in slices:
            ratings.append(slice_rating[slice])
        ratings = numpy.asarray(ratings, dtype=theano.config.floatX)
        if dyad in dyad_features:
            dyad_features[dyad] = numpy.concatenate((dyad_features[dyad], X), axis=0)
            dyad_ratings[dyad] = numpy.concatenate((dyad_ratings[dyad], ratings), axis=0)
            for slice in slices:
                dyad_slices[dyad].append(slice)
        else:
            dyad_features[dyad] = X
            dyad_ratings[dyad] = ratings
            dyad_slices[dyad] = slices
    return dyad_features, dyad_ratings, dyad_slices


def load_dyad_audio(dirname, side='b', num_frame=300, valid_slices=None, normalization=True):
    slice_features = {}

    files = os.listdir(dirname)
    files.sort()
    for mat_name in files:
        if not mat_name.endswith('mat'):
            continue
        sp = mat_name.split('_')
        dyad = int(sp[0][1:])
        session = int(sp[1][1:])
        slice = int(sp[2])
        if valid_slices is not None and str(dyad) + '_' + str(session) + '_' + str(slice) not in valid_slices:
            continue
        lr = sp[3][:-4]
        mat_path = os.path.join(dirname, mat_name)
        data = loadmat(mat_path)
        feat = data['features']
        if feat.shape[0] < 2500:
            continue
        '''
        interval = feat.shape[0] // num_frame
        index = ind * interval
        feat = feat[index]'''

        interval = feat.shape[0] // num_frame
        feat = feat[:interval * num_frame]
        feat = numpy.reshape(feat, (num_frame, interval, feat.shape[1]))
        feat = numpy.mean(feat, axis=1)

        feat[numpy.isneginf(feat)] = -1.

        if normalization:
            feat = normalize(feat, norm='l1', axis=0)
            feat = normalize(feat)

        slice_tup = (dyad, session, slice)
        if slice_tup not in slice_features:
            slice_features[slice_tup] = {lr: feat}
        else:
            slice_features[slice_tup][lr] = feat
    slices = []
    features = []
    if side == 'lr':
        for slice, feats in slice_features.items():
            slices.append(slice)
            features.append(feats['left'])
            slices.append(slice)
            features.append(feats['right'])
    elif side == 'l':
        for slice, feats in slice_features.items():
            slices.append(slice)
            features.append(feats['left'])
    elif side == 'r':
        for slice, feats in slice_features.items():
            slices.append(slice)
            features.append(feats['right'])
    elif side == 'ba':
        for slice, feats in slice_features.items():
            slices.append(slice)
            features.append(feats['left'] + feats['right'])
    else:
        for slice, feats in slice_features.items():
            slices.append(slice)
            features.append(numpy.concatenate((feats['left'], feats['right']), axis=1))
            slices.append(slice)
            features.append(numpy.concatenate((feats['right'], feats['left']), axis=1))
    if len(features) == 0:
        return None, None
    X = numpy.stack(features, axis=0).astype(theano.config.floatX)

    '''
    X_tmp = numpy.reshape(X, (X.shape[0] * X.shape[1], X.shape[2]))
    X_tmp = StandardScaler().fit_transform(X_tmp)
    X = numpy.reshape(X_tmp, (X.shape[0], X.shape[1], X.shape[2]))'''

    return X, slices


def test1():
    load_vision(sample_10_root)


def test2():
    load_audio()


def test3():
    slice_rating = load_ratings(category=True, best3=False)
    count = [0, 0, 0]
    for rating in slice_rating.values():
        count[rating] += 1
    s = sum(count)
    for i in xrange(3):
        count[i] /= float(s)
    print count


if __name__ == '__main__':
    test3()
