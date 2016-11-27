__author__ = 'yuhongliang324'
import numpy
import theano


n_class = 7


def load():
    X = numpy.random.rand(2000, 50, 1000).astype(theano.config.floatX)
    X.transpose([1, 0, 2])
    print X.shape
    y = numpy.random.randint(0, n_class, (2000,)).astype(theano.config.floatX)
    return X, y


def load_dyad(dirname, binarize=True):
    pass
