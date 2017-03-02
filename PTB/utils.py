__author__ = 'yuhongliang324'
import os
import numpy
import theano
import cPickle


data_root = 'processed_data/'
train_file = os.path.join(data_root, 'train.txt')
valid_file = os.path.join(data_root, 'dev.txt')
test_file = os.path.join(data_root, 'test.txt')
wordvec_file = '/usr0/home/hongliay/word_vectors/glove.840B.300d.txt'
dict_pkl = os.path.join(data_root, 'token_vec.pkl')

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


def test1():
    tokens = get_dict()
    print len(tokens)
    get_vectors(tokens)


if __name__ == '__main__':
    test1()


