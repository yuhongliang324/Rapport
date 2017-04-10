__author__ = 'yuhongliang324'
import os
from nltk import word_tokenize
import sys
sys.path.append('..')
from SST import utils as SU
import cPickle
import numpy
import theano
from collections import defaultdict
import random


origin_data_root = '/usr0/home/hongliay/datasets/aclImdb/'
origin_train_pos_root = os.path.join(origin_data_root, 'train/pos/')
origin_train_neg_root = os.path.join(origin_data_root, 'train/neg/')
origin_test_pos_root = os.path.join(origin_data_root, 'test/pos/')
origin_test_neg_root = os.path.join(origin_data_root, 'test/neg/')

POSITIVE = 0
NEGATIVE = 1
num_class = 2

data_root = os.path.join(origin_data_root, 'processed_data/')
train_file = os.path.join(data_root, 'train.txt')
test_file = os.path.join(data_root, 'test.txt')

dict_pkl = os.path.join(data_root, 'dict.pkl')

train_pkl = os.path.join(data_root, 'train.pkl')
test_pkl = os.path.join(data_root, 'test.pkl')


def process_to_single_file(pos_path, neg_path, out_file):
    writer = open(out_file, 'w')

    def write_file(dir_path):
        files = os.listdir(dir_path)
        num_files = len(files)
        files.sort()
        for i, fn in enumerate(files):
            if not fn.endswith('txt'):
                continue
            if (i + 1) % 1000 == 0:
                print i + 1, '/', num_files
            label = int(fn.split('.')[0].split('_')[1])
            fp = os.path.join(dir_path, fn)
            reader = open(fp)
            lines = reader.readlines()
            reader.close()
            line = lines[0].strip().lower()
            line = line.replace('<br />', ' ')
            tokens = word_tokenize(line)
            line = ' '.join(tokens)
            writer.write(str(label) + ' ' + line + '\n')

    write_file(pos_path)
    write_file(neg_path)
    writer.close()


def get_dict():
    tokens = set()

    def get_dict1(fn):
        reader = open(fn)
        lines = reader.readlines()
        reader.close()
        lines = map(lambda x: x.strip(), lines)
        for line in lines:
            words = line.split()
            num_words = len(words)
            for i in xrange(1, num_words):
                tokens.add(words[i])
                if '-' in words[i]:
                    for w in words[i].split('-'):
                        tokens.add(w)
    get_dict1(train_file)
    get_dict1(test_file)

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


def load_dict(vec_file=dict_pkl):
    reader = open(vec_file, 'rb')
    token_ID, E = cPickle.load(reader)
    reader.close()
    return token_ID, E.astype(theano.config.floatX)


def vectorize_data(file_name, token_ID, out_file):
    reader = open(file_name)
    lines = reader.readlines()
    reader.close()
    lines = map(lambda x: x.strip(), lines)
    xs = []
    ys = []
    for line in lines:
        sp = line.split()
        y = int(sp[0])
        x = []
        for tok in sp[1:]:
            x.append(token_ID[tok])
        xs.append(x)
        ys.append(y)

    f = open(out_file, 'wb')
    cPickle.dump([xs, ys], f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()

    return xs, ys


def sort_by_length(Xs, ys, indices):
    def bylen(a, b):
        if a[-2] != b[-2]:
            return a[-2] - b[-2]
        return a[-1] - b[-1]
    lens = [len(vec) for vec in Xs]
    num_sent = len(ys)
    ind = range(num_sent)
    random.shuffle(ind)
    cb = zip(Xs, ys, indices, lens, ind)
    cb.sort(cmp=bylen)
    Xs = [item[0] for item in cb]
    ys = [item[1] for item in cb]
    indices = [item[2] for item in cb]
    return Xs, ys, indices


def load_data(pkl_file, batch_size=100, binary=True):
    reader = open(pkl_file)
    [xs, ys] = cPickle.load(reader)
    n = len(xs)
    if binary:
        for i in xrange(n):
            if ys[i] < 5:
                ys[i] = NEGATIVE
            else:
                ys[i] = POSITIVE
    reader.close()
    indices = numpy.arange(len(xs)).tolist()
    xs, ys, indices = sort_by_length(xs, ys, indices)
    lengths = [len(x) for x in xs]


    num_batch = (n + batch_size - 1) // batch_size
    start_batches, end_batches, len_batches = [], [], []
    xs_short = []
    for i in xrange(num_batch):
        start, end = i * batch_size, min((i + 1) * batch_size, n)
        length = lengths[start]
        for j in xrange(start, end):
            dif = (lengths[j] - length) // 2
            xs_short.append(xs[j][dif: dif + length])
        start_batches.append(start)
        end_batches.append(end)
        len_batches.append(length)

    # Pad xs_short
    maxLen = len_batches[-1]
    X = numpy.zeros((n, maxLen), dtype='int32') - 1
    for i in xrange(num_batch):
        start, end = start_batches[i], end_batches[i]
        length = len_batches[i]
        for j in xrange(start, end):
            X[j, :length] = numpy.asarray(xs_short[j], dtype='int32')
    y = numpy.asarray(ys, dtype='int32')

    # shuffle start_batches, end_batches, len_batches
    z = zip(start_batches, end_batches, len_batches)
    random.shuffle(z)
    start_batches = [item[0] for item in z]
    start_batches = numpy.asarray(start_batches, dtype='int32')
    end_batches = [item[1] for item in z]
    end_batches = numpy.asarray(end_batches, dtype='int32')
    len_batches = [item[2] for item in z]
    len_batches = numpy.asarray(len_batches, dtype='int32')

    return X, y, start_batches, end_batches, len_batches, indices


def test1():
    process_to_single_file(origin_train_pos_root, origin_train_neg_root, train_file)
    process_to_single_file(origin_test_pos_root, origin_test_neg_root, test_file)


def test2():
    tokens = get_dict()
    print len(tokens)
    get_vectors(tokens, out_file=dict_pkl)


def test3():
    token_ID, _ = load_dict()
    vectorize_data(train_file, token_ID, train_pkl)
    vectorize_data(test_file, token_ID, test_pkl)


if __name__ == '__main__':
    test3()
