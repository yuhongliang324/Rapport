__author__ = 'yuhongliang324'
import os
import cPickle as pickle
from scipy.io import loadmat, savemat
import numpy
from SST import utils as SU
import theano
from collections import defaultdict
import cPickle


raw_data_root = '/multicomp/datasets/mosi'
raw_openface_pkl = os.path.join(raw_data_root, 'OpenFaceFeatures.pkl')
raw_audio_root = os.path.join(raw_data_root, 'Audio/segment')
raw_facet_root = os.path.join(raw_data_root, 'FACET_GIOTA')
raw_hog_root = os.path.join(raw_data_root, 'HOG/PCA_HOG')
raw_text_root = os.path.join(raw_data_root, 'text_segment_aligned')

data_root = '/multicomp/datasets/mosi/Features'
data_root_mat = '/multicomp/datasets/mosi/Features_mat'
split_file = os.path.join(raw_data_root, 'split.txt')

preprocessed_root = 'preprocessed'
dict_pkl = os.path.join(preprocessed_root, 'dict.pkl')


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
    print 'Audio features loaded'
    return data


def get_hog_features():
    data = {}
    mats = os.listdir(raw_hog_root)
    mats.sort()
    for mat in mats:
        if not mat.endswith('mat'):
            continue
        mat_path = os.path.join(raw_hog_root, mat)
        d = loadmat(mat_path)
        feat = d['hog_feature']
        videoID = mat[:11]
        segID = mat[12:-4]
        if videoID not in data:
            data[videoID] = {}
        data[videoID][segID] = feat
    print 'HOG features loaded'
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
    print 'Facet features loaded'
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


# ---for text feature embedding extraction-----------------------

def get_vocabulary():
    files = os.listdir(raw_text_root)
    files.sort()
    tokens = set()
    for fn in files:
        if not (len(fn) == 13 or len(fn) == 14):
            continue
        reader = open(os.path.join(raw_text_root, fn))
        lines = reader.readlines()
        reader.close()
        lines = map(lambda x: x.strip().split(',')[1].lower(), lines)
        sent = ' '.join(lines)
        words = sent.split()
        for word in words:
            tokens.add(word)
    print len(tokens)
    return tokens


def get_vectors(tokens, vec_file=SU.wordvec_file, out_file=dict_pkl):
    token_vec = {}
    reader = open(vec_file)
    count = 0
    while True:
        line = reader.readline()
        if line:
            count += 1
            if count % 100000 == 0:
                print count
            line = line.strip()
            sp = line.split()
            sp[0].replace('\'', '').replace('-', '')
            if sp[0] not in tokens:
                continue
            tok = sp[0]
            vec = [float(x) for x in sp[1:]]
            vec = numpy.asarray(vec, dtype=theano.config.floatX)
            token_vec[tok] = vec
        else:
            break
    reader.close()
    V = len(token_vec)
    print V

    E = numpy.zeros((V + 1, token_vec['the'].shape[0]))
    token_ID = defaultdict(int)
    curID = 1

    for token, vec in token_vec.items():
        token_ID[token] = curID
        E[curID, :] = vec
        curID += 1
    E[0, :] = numpy.mean(E[1:], axis=0)

    f = open(out_file, 'wb')
    cPickle.dump([token_ID, E], f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()


def get_sentence_vectors(token_ID, E):
    files = os.listdir(raw_text_root)
    files.sort()
    data = {}
    unknown = set()
    for fn in files:
        if not (len(fn) == 13 or len(fn) == 14):
            continue
        videoID = fn[: 11]
        segID = fn[12:]
        if videoID not in data:
            data[videoID] = {}
        reader = open(os.path.join(raw_text_root, fn))
        lines = reader.readlines()
        reader.close()
        lines = map(lambda x: x.strip().split(',')[1].lower(), lines)
        sent = ' '.join(lines)
        tokens = sent.split()
        tokenIDs = []
        for tok in tokens:
            if tok not in token_ID:
                tokenIDs.append(0)
                unknown.add(tok)
            else:
                tokenIDs.append(token_ID[tok])
        data[videoID][segID] = E[tokenIDs]
        print len(tokens), E[tokenIDs].shape
        print unknown
    return data

# ---------------------------------------------------------------


def write_features_mat(data_openface, data_audio, data_facet):
    videoIDs = data_openface.keys()
    for videoID in videoIDs:
        print videoID
        segID_cont = data_openface[videoID]
        for segID, cont in segID_cont.items():
            new_cont = {'start_time': cont['start_time'], 'end_time': cont['end_time'], 'label': cont['sentiment']}
            feat_openface = cont['openface']
            new_cont['openface'] = numpy.asarray(feat_openface, dtype=numpy.float32)
            new_cont['facet'] = numpy.asarray(data_facet[videoID][segID], dtype=numpy.float32)
            new_cont['audio'] = numpy.asarray(data_audio[videoID][segID], dtype=numpy.float32)
            seg_path = os.path.join(data_root_mat, videoID + '_' + segID + '.mat')
            savemat(seg_path, new_cont)


def split_data():
    files = os.listdir(data_root)
    files.sort()
    files = map(lambda x: x[:-4], files)
    total = len(files)
    num_test = total // 3
    num_train = total - num_test
    test_set = []
    train_set = []

    for i in xrange(total):
        if i < num_train:
            train_set.append(files[i])
        else:
            test_set.append(files[i])
    print len(train_set), len(test_set)

    writer = open(split_file, 'w')
    writer.write(' '.join(test_set) + '\n')
    writer.write(' '.join(train_set) + '\n')


def test1():
    data_openface = get_openface_features()
    data_audio = get_audio_features()
    data_facet = get_facet_features(data_openface)
    write_features(data_openface, data_audio, data_facet)


def test2():
    split_data()


def test3():
    get_vocabulary()


if __name__ == '__main__':
    test3()
