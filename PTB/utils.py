__author__ = 'yuhongliang324'
import os
import numpy
import theano
import cPickle


data_root = 'processed_data/'
train_file = os.path.join(data_root, 'train.txt')
valid_file = os.path.join(data_root, 'dev.txt')
test_file = os.path.join(data_root, 'test.txt')

train_pkl = os.path.join(data_root, 'train.pkl')
valid_pkl = os.path.join(data_root, 'dev.pkl')
test_pkl = os.path.join(data_root, 'test.pkl')

wordvec_file = '/usr0/home/hongliay/word_vectors/glove.840B.300d.txt'

dict_pkl = os.path.join(data_root, 'token_vec.pkl')

UNKNOWN = '*UNKNOWN*'


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
                if words[i] == '-lrb-':
                    words[i] = '('
                elif words[i] == '-rrb-':
                    words[i] = ')'
                tokens.add(words[i])
                if '-' in words[i]:
                    for w in words[i].split('-'):
                        tokens.add(w)
    get_dict1(train_file)
    get_dict1(valid_file)
    get_dict1(test_file)

    return tokens


def get_vectors(tokens, vec_file=wordvec_file, out_file=dict_pkl):
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
    print len(token_vec)
    print token_vec['the'].shape
    f = open(out_file, 'wb')
    cPickle.dump(token_vec, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()
    return token_vec


def load_dict(vec_file=dict_pkl):
    reader = open(vec_file, 'rb')
    token_vec = cPickle.load(reader)
    reader.close()
    vecs = token_vec.values()
    st = numpy.stack(vecs)
    unk_vec = numpy.mean(st, axis=0)
    token_vec[UNKNOWN] = unk_vec
    return token_vec


def vectorize_data(file_name, token_vec, out_file):
    reader = open(file_name)
    lines = reader.readlines()
    reader.close()
    lines = map(lambda x: x.strip(), lines)
    Xs = []
    ys = []
    for line in lines:
        sp = line.split()
        y = int(sp[0])
        X = []
        for tok in sp[1:]:
            if tok == '-lrb-':
                tok = '('
            elif tok == '-rrb-':
                tok = ')'
            if tok in token_vec:
                X.append(token_vec[tok])
            elif '-' in tok:
                ts = tok.split('-')
                x_tmp = []
                for t in ts:
                    if t in token_vec:
                        x_tmp.append(token_vec[t])
                    else:
                        x_tmp.append(token_vec[UNKNOWN])
                x_tmp = numpy.stack(x_tmp)
                x_tmp = numpy.mean(x_tmp, axis=0)
                X.append(x_tmp)
            else:
                X.append(token_vec[tok])
        Xs.append(X)
        ys.append(y)
    ys = numpy.asarray(ys, dtype=theano.config.floatX)

    f = open(out_file, 'wb')
    cPickle.dump([Xs, ys], f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()

    return Xs, ys


def test1():
    tokens = get_dict()
    print len(tokens)
    get_vectors(tokens)


def test2():
    token_vec = load_dict()
    vectorize_data(train_file, token_vec, train_pkl)
    vectorize_data(valid_file, token_vec, valid_pkl)
    vectorize_data(test_file, token_vec, test_pkl)


if __name__ == '__main__':
    test2()


