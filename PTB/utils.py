__author__ = 'yuhongliang324'
import os
import numpy
import theano


data_root = 'processed_data/'
train_file = os.path.join(data_root, 'train.txt')
valid_file = os.path.join(data_root, 'dev.txt')
test_file = os.path.join(data_root, 'test.txt')
wordvec_file = '/usr0/home/hongliay/word_vectors/glove.840B.300d.txt'


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


def get_vectors(tokens, vec_file=wordvec_file):
    token_vec = {}
    reader = open(vec_file)
    count = 0
    while True:
        line = reader.readline()
        if line:
            line = line.strip()
            sp = line.split()
            if sp[0] not in tokens:
                continue
            tok = sp[0]
            vec = [float(x) for x in sp[1:]]
            vec = numpy.asarray(vec, dtype=theano.config.floatX)
            token_vec[tok] = vec
            count += 1
            if count % 10000 == 0:
                print count
        else:
            break
    print len(token_vec)
    return token_vec


def test1():
    tokens = get_dict()
    print len(tokens)
    get_vectors(tokens)


if __name__ == '__main__':
    test1()


