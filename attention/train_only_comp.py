__author__ = 'yuhongliang324'

# The training and testing are comparisons

from math import ceil, sqrt
import numpy
import theano
import theano.tensor as T

from comp_net import ComparisonNet
import sys
sys.path.append('../')


def validate(test_model, y_test, batch_size=32):
    n_test = y_test.shape[0]
    num_iter = int(ceil(n_test / float(batch_size)))
    cost, acc = 0, 0
    for iter_index in xrange(num_iter):
        start, end = iter_index * batch_size, min((iter_index + 1) * batch_size, n_test)
        cost_iter, acc_iter = test_model(start, end)
        cost += cost_iter * (end - start)
        acc += acc * (end - start)
    cost /= n_test
    acc /= n_test
    print '\tTest cost = %f,\tAccuracy = %f' % (cost, acc)


def train(X1_train, X2_train, gap_train, X1_test, X2_test, y_test, n_class=1, hidden_dim=128, batch_size=64, num_epoch=2):

    n_train = X1_train.shape[0]
    input_dim = X1_train.shape[2]
    X1_train = X1_train.transpose([1, 0, 2])
    X2_train = X2_train.transpose([1, 0, 2])
    X1_test = X1_test.transpose([1, 0, 2])
    X2_test = X2_test.transpose([1, 0, 2])

    X1_train_shared = theano.shared(X1_train, borrow=True)
    X2_train_shared = theano.shared(X2_train, borrow=True)
    y_train_shared = T.cast(theano.shared(gap_train, borrow=True), 'int32')
    X1_test_shared = theano.shared(X1_test, borrow=True)
    X2_test_shared = theano.shared(X2_test, borrow=True)
    y_test_shared = T.cast(theano.shared(y_test, borrow=True), 'int32')

    cn = ComparisonNet(input_dim, hidden_dim, n_class=n_class)

    symbols = cn.build_model()

    X1_batch, X2_batch, y_batch = symbols['X1_batch'], symbols['X2_batch'], symbols['y_batch']
    cost, acc, updates = symbols['cost'], symbols['acc'], symbols['updates']

    num_iter = int(ceil(n_train / float(batch_size)))

    print 'Compiling function'

    start_symbol, end_symbol = T.lscalar(), T.lscalar()

    train_model = theano.function(inputs=[start_symbol, end_symbol],
                                  outputs=[cost, acc], updates=updates,
                                  givens={
                                      X1_batch: X1_train_shared[:, start_symbol: end_symbol, :],
                                      X2_batch: X2_train_shared[:, start_symbol: end_symbol, :],
                                      y_batch: y_train_shared[start_symbol: end_symbol]},
                                  on_unused_input='ignore')
    test_model = theano.function(inputs=[start_symbol, end_symbol],
                                  outputs=[cost, acc],
                                  givens={
                                      X1_batch: X1_test_shared[:, start_symbol: end_symbol, :],
                                      X2_batch: X2_test_shared[:, start_symbol: end_symbol, :],
                                      y_batch: y_test_shared[start_symbol: end_symbol]},
                                  on_unused_input='ignore')
    print 'Compilation done'

    for epoch_index in xrange(num_epoch):
        cost, acc = 0., 0.
        print 'Epoch = %d' % (epoch_index + 1)
        for iter_index in xrange(num_iter):
            start, end = iter_index * batch_size, min((iter_index + 1) * batch_size, n_train)
            cost_iter, acc_iter = train_model(start, end)
            cost += cost_iter * (end - start)
            acc += acc_iter * (end - start)
        cost /= n_train
        print '\tTrain cost = %f,\tAccuracy = %f' % (cost, acc)
        validate(test_model, y_test)


def cross_validation(n_class):
    from data_preprocessing.load_data import load_pairs
    from data_path import sample_10_root
    print 'Preparing pairs ... '
    dyad_X1, dyad_X2, dyad_gaps = load_pairs(sample_10_root, n_class=n_class)
    dyads = dyad_X1.keys()
    num_dyad = len(dyads)
    for i in xrange(num_dyad):  # num_dyad
        dyad = dyads[i]
        X1_test = dyad_X1[dyad]
        X2_test = dyad_X2[dyad]
        y_test = dyad_gaps[dyad]
        X1_list, X2_list, rating_list = [], [], []
        for j in xrange(num_dyad):
            if j == i:
                continue
            X1_list.append(dyad_X1[dyads[j]])
            X2_list.append(dyad_X2[dyads[j]])
            rating_list.append(dyad_gaps[dyads[j]])
        X1_train = numpy.concatenate(X1_list)
        X2_train = numpy.concatenate(X2_list)
        gap_train = numpy.concatenate(rating_list)

        # shuffle
        print 'Shuffling ... '
        indices = numpy.arange(X1_train.shape[0])
        numpy.random.shuffle(indices)
        X1_train = X1_train[indices]
        X2_train = X2_train[indices]
        gap_train = gap_train[indices]

        print X1_train.shape, X2_train.shape, X1_test.shape, X2_test.shape

        train(X1_train, X2_train, gap_train, X1_test, X2_test, y_test, n_class=n_class, hidden_dim=128)


def test1():
    cross_validation()


if __name__ == '__main__':
    test1()


