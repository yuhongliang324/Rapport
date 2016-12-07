__author__ = 'yuhongliang324'

from math import ceil, sqrt
import numpy
import theano
import theano.tensor as T
from sklearn.metrics import mean_squared_error

from rnn_attention import RNN_Attention
import sys
sys.path.append('../')


def RMSE(y_actual, y_predicted):
    rmse = sqrt(mean_squared_error(y_actual, y_predicted))
    return rmse


def validate(test_model, y_test, batch_size=16):
    n_test = y_test.shape[0]
    print n_test
    num_iter = int(ceil(n_test / float(batch_size)))
    all_pred = []
    cost_avg = 0.
    for iter_index in xrange(num_iter):
        start, end = iter_index * batch_size, min((iter_index + 1) * batch_size, n_test)
        cost, pred = test_model(start, end)
        cost_avg += cost * (end - start)
        all_pred += pred.tolist()
    cost_avg /= n_test
    rmse = RMSE(y_test, all_pred)
    print '\tTest cost = %f,\tRMSE = %f' % (cost_avg, rmse)


def train(X_train, y_train, X_test, y_test, hidden_dim=512, batch_size=16, num_epoch=40):

    n_train = X_train.shape[0]
    input_dim = X_train.shape[2]
    X_train = X_train.transpose([1, 0, 2])
    X_test = X_test.transpose([1, 0, 2])

    X_train_shared = theano.shared(X_train, borrow=True)
    y_train_shared = theano.shared(y_train, borrow=True)
    X_test_shared = theano.shared(X_test, borrow=True)
    y_test_shared = theano.shared(numpy.zeros_like(y_test, dtype=theano.config.floatX), borrow=True)

    ra = RNN_Attention(input_dim, hidden_dim, 1)
    symbols = ra.build_model()

    X_batch, y_batch = symbols['X_batch'], symbols['y_batch']
    att, pred, loss = symbols['a'], symbols['pred'], symbols['loss']
    cost, updates = symbols['cost'], symbols['updates']

    num_iter = int(ceil(n_train / float(batch_size)))

    print 'Compiling function'

    start_symbol, end_symbol = T.lscalar(), T.lscalar()

    train_model = theano.function(inputs=[start_symbol, end_symbol],
                                  outputs=[cost, pred], updates=updates,
                                  givens={
                                      X_batch: X_train_shared[:, start_symbol: end_symbol, :],
                                      y_batch: y_train_shared[start_symbol: end_symbol]},
                                  on_unused_input='ignore')
    test_model = theano.function(inputs=[start_symbol, end_symbol],
                                  outputs=[cost, pred],
                                  givens={
                                      X_batch: X_test_shared[:, start_symbol: end_symbol, :],
                                      y_batch: y_test_shared[start_symbol: end_symbol]},
                                  on_unused_input='ignore')
    print 'Compilation done'

    for epoch_index in xrange(num_epoch):
        cost_avg, rmse = 0., 0.
        all_pred = []
        print 'Epoch = %d' % (epoch_index + 1)
        for iter_index in xrange(num_iter):
            start, end = iter_index * batch_size, min((iter_index + 1) * batch_size, n_train)
            cost, pred = train_model(start, end)
            cost_avg += cost * (end - start)
            all_pred += pred.tolist()
        cost_avg /= n_train
        y_predicted = numpy.asarray(all_pred)
        rmse = RMSE(y_train, y_predicted)
        print '\tTrain cost = %f,\tRMSE = %f' % (cost_avg, rmse)
        validate(test_model, y_test)


def cross_validation():
    from data_preprocessing.load_data import load
    from data_path import sample_10_root
    dyad_features, dyad_ratings = load(sample_10_root)
    dyads = dyad_features.keys()
    num_dyad = len(dyads)
    for i in xrange(num_dyad):
        dyad = dyads[i]
        X_test = dyad_features[dyad]
        y_test = dyad_ratings[dyad]
        feature_list, rating_list = [], []
        for j in xrange(num_dyad):
            if j == i:
                continue
            feature_list.append(dyad_features[dyads[i]])
            rating_list.append(dyad_ratings[dyads[i]])
        X_train = numpy.concatenate(feature_list)
        y_train = numpy.concatenate(rating_list)
        rating_mean = numpy.mean(y_train)
        rmse = y_test - rating_mean
        rmse = sqrt(numpy.mean(rmse * rmse))
        print 'Testing Dyad =', dyad
        print 'RMSE of Average Prediction = %f' % rmse
        print X_train.shape, X_test.shape
        X_test = X_train[:X_test.shape[0]]
        print X_test.shape
        train(X_train, y_train, X_test, y_test)


def test1():
    cross_validation()


if __name__ == '__main__':
    test1()
