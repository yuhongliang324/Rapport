__author__ = 'yuhongliang324'
import os
from scipy.io import loadmat
import numpy
import random
import theano
from sklearn.preprocessing import normalize


data_root = '/multicomp/users/liangke/MM/datasets/IEMOCAP/features'
# Order: Angry, Happy, Sad, Neutral
oldl_newl = {2: 0, 8: 1, 5: 2, 9: 3, 3: 1}


def load_feature(feature_name='audio'):
    sessions = os.listdir(data_root)
    sessions.sort()
    session_Xs = {}
    session_y = {}
    for session in sessions:
        if not session.startswith('Session'):
            continue
        print session
        sessionID = int(session[-1])
        session_path = os.path.join(data_root, session)
        Xs, y = load_feature_session(session_path, feature_name)
        session_Xs[sessionID] = Xs
        session_y[sessionID] = y
    print 'Features loaded'
    return session_Xs, session_y


def load_feature_session(session_path, feature_name):
    Xs = []
    y = []
    videos = os.listdir(session_path)
    videos.sort()
    for video in videos:
        video_path = os.path.join(session_path, video)
        if not os.path.isdir(video_path):
            continue
        mats = os.listdir(video_path)
        mats.sort()
        for mat in mats:
            if not mat.endswith('mat'):
                continue
            if mat == 'Ses04F_script02_1_M039.mat':
                continue
            mat_path = os.path.join(video_path, mat)
            data = loadmat(mat_path)
            label = data['emotion_label'][0][0]
            if label not in oldl_newl:
                continue
            label = oldl_newl[label]
            if feature_name == 'audio':
                X = data['audio_features']
            else:
                X = data['rotated_features']
                X = normalize(X, norm='l1', axis=0)  # !!!
                X = normalize(X)  # !!!
            X[numpy.isneginf(X)] = -1.
            X[numpy.isnan(X)] = 0.
            Xs.append(X)
            y.append(label)
    return Xs, y


# Fill whether maxlen or sample_rate
def subsample(X, average=False, maxlen=None, sample_rate=None):
    if maxlen is not None:
        sample_rate = (X.shape[0] + maxlen - 1) // maxlen
    if average:
        num_frame = X.shape[0] // sample_rate
        X = X[:sample_rate * num_frame]
        X = numpy.reshape(X, (num_frame, sample_rate, X.shape[1]))
        X = numpy.mean(X, axis=1)
    else:
        num_frame = X.shape[0] // sample_rate
        ind = numpy.arange(num_frame) * sample_rate
        X = X[ind]
    return X


def sort_by_length(Xs, ys):
    def bylen(a, b):
        if a[2] != b[2]:
            return a[2] - b[2]
        return a[3] - b[3]
    lens = [X.shape[0] for X in Xs]
    num_sent = len(ys)
    ind = range(num_sent)
    random.shuffle(ind)
    cb = zip(Xs, ys, lens, ind)
    cb.sort(cmp=bylen)
    Xs = [item[0] for item in cb]
    ys = [item[1] for item in cb]
    return Xs, ys


def pad(X_list, y_list, batch_size=16, average=False, maxlen=1000, sample_rate=5):
    size = len(X_list)

    # Subsampling
    for i in xrange(size):
        X = X_list[i]
        if X.shape[0] > maxlen:
            X = subsample(X, average=average, maxlen=maxlen)
        X = subsample(X, average=average, sample_rate=sample_rate)
        X_list[i] = X

    X_list, y_list = sort_by_length(X_list, y_list)

    num_batch = (size + batch_size - 1) // batch_size
    start_batches, end_batches, len_batches = [], [], []
    Xs_short = []
    for i in xrange(num_batch):
        start, end = i * batch_size, min((i + 1) * batch_size, size)
        length = X_list[start].shape[0]
        for j in xrange(start, end):
            dif = (X_list[j].shape[0] - length) // 2
            Xs_short.append(X_list[j][dif: dif + length])
        start_batches.append(start)
        end_batches.append(end)
        len_batches.append(length)

    # Pad Xs_short
    maxLen = len_batches[-1]
    Xs = numpy.zeros((size, maxLen, X_list[0].shape[-1]), dtype=theano.config.floatX)
    for i in xrange(num_batch):
        start, end = start_batches[i], end_batches[i]
        length = len_batches[i]
        for j in xrange(start, end):
            Xs[j, :length, :] = Xs_short[j]
    y = numpy.asarray(y_list, dtype='int32')

    # Shuffle start_batches, end_batches, len_batches
    z = zip(start_batches, end_batches, len_batches)
    random.shuffle(z)
    start_batches = [item[0] for item in z]
    start_batches = numpy.asarray(start_batches, dtype='int32')
    end_batches = [item[1] for item in z]
    end_batches = numpy.asarray(end_batches, dtype='int32')
    len_batches = [item[2] for item in z]
    len_batches = numpy.asarray(len_batches, dtype='int32')

    return Xs, y, start_batches, end_batches, len_batches


def test1():
    load_feature(feature_name='video')


if __name__ == '__main__':
    test1()
