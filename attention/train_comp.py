__author__ = 'yuhongliang324'

from math import ceil, sqrt
import numpy
import theano
import theano.tensor as T
from sklearn.metrics import mean_squared_error

from comp_net import ComparisonNet
import sys
sys.path.append('../')


def RMSE(y_actual, y_predicted):
    rmse = sqrt(mean_squared_error(y_actual, y_predicted))
    return rmse


def validate(test_model, y_test, y_mean, batch_size=32):
    n_test = y_test.shape[0]
    num_iter = int(ceil(n_test / float(batch_size)))
    all_pred = []
    for iter_index in xrange(num_iter):
        start, end = iter_index * batch_size, min((iter_index + 1) * batch_size, n_test)
        [pred] = test_model(start, end)
        all_pred += pred.tolist()
    all_pred = numpy.asarray(all_pred)
    all_pred += y_mean - numpy.mean(all_pred)
    all_pred = all_pred.tolist()
    rmse = RMSE(y_test, all_pred)
    print '\tTest cost = NAN,\tRMSE = %f' % rmse


def train(X1_train, X2_train, gap_train, X_test, y_test, y_mean, hidden_dim=128, batch_size=64, num_epoch=2):

    n_train = X1_train.shape[0]
    input_dim = X1_train.shape[2]
    X1_train = X1_train.transpose([1, 0, 2])
    X2_train = X2_train.transpose([1, 0, 2])
    X_test = X_test.transpose([1, 0, 2])

    X1_train_shared = theano.shared(X1_train, borrow=True)
    X2_train_shared = theano.shared(X2_train, borrow=True)
    y_train_shared = theano.shared(gap_train, borrow=True)
    X_test_shared = theano.shared(X_test, borrow=True)
    y_test_shared = theano.shared(y_test, borrow=True)

    cn = ComparisonNet(input_dim, hidden_dim)

    symbols = cn.build_model()

    X1_batch, X2_batch, y_batch = symbols['X1_batch'], symbols['X2_batch'], symbols['y_batch']
    pred1 = symbols['pred1']
    cost, updates = symbols['cost'], symbols['updates']

    num_iter = int(ceil(n_train / float(batch_size)))

    print 'Compiling function'

    start_symbol, end_symbol = T.lscalar(), T.lscalar()

    train_model = theano.function(inputs=[start_symbol, end_symbol],
                                  outputs=[cost], updates=updates,
                                  givens={
                                      X1_batch: X1_train_shared[:, start_symbol: end_symbol, :],
                                      X2_batch: X2_train_shared[:, start_symbol: end_symbol, :],
                                      y_batch: y_train_shared[start_symbol: end_symbol]},
                                  on_unused_input='ignore')
    test_model = theano.function(inputs=[start_symbol, end_symbol],
                                  outputs=[pred1],
                                  givens={
                                      X1_batch: X_test_shared[:, start_symbol: end_symbol, :],
                                      y_batch: y_test_shared[start_symbol: end_symbol]},
                                  on_unused_input='ignore')
    print 'Compilation done'

    for epoch_index in xrange(num_epoch):
        cost_avg, rmse = 0., 0.
        print 'Epoch = %d' % (epoch_index + 1)
        for iter_index in xrange(num_iter):
            start, end = iter_index * batch_size, min((iter_index + 1) * batch_size, n_train)
            [cost] = train_model(start, end)
            cost_avg += cost * (end - start)
        cost_avg /= n_train
        print '\tTrain cost = %f,\tRMSE = NAN' % cost_avg
        validate(test_model, y_test, y_mean)


def cross_validation():
    from data_preprocessing.load_data import load, load_pairs
    from data_path import sample_10_root
    print 'Preparing normal features ... '
    dyad_features, dyad_ratings = load(sample_10_root)
    print 'Preparing pairs ... '
    dyad_X1, dyad_X2, dyad_gaps = load_pairs(sample_10_root)
    dyads = dyad_X1.keys()
    num_dyad = len(dyads)
    for i in xrange(num_dyad):  # num_dyad
        dyad = dyads[i]
        X_test = dyad_features[dyad]
        y_test = dyad_ratings[dyad]
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

        print X1_train.shape, X2_train.shape, X_test.shape

        y_mean = 0.
        for j in xrange(num_dyad):
            if j == i:
                continue
            y_mean += numpy.mean(dyad_ratings[dyads[j]])
        y_mean /= num_dyad - 1

        train(X1_train, X2_train, gap_train, X_test, y_test, y_mean, hidden_dim=128)


def test1():
    cross_validation()


if __name__ == '__main__':
    test1()

