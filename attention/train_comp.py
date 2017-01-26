__author__ = 'yuhongliang324'

from math import ceil
import numpy
import theano
import theano.tensor as T

from rnn_attention import RNN_Attention
from bi_rnn_attention import Bi_RNN_Attention
import sys
sys.path.append('../')
from model_utils import plot_loss
import argparse
import random


def validate(test_model, y_test, costs_val, accs_val, batch_size=32):
    n_test = y_test.shape[0]
    num_iter = int(ceil(n_test / float(batch_size)))
    cost_avg = 0.
    acc_avg = 0.
    for iter_index in xrange(num_iter):
        start, end = iter_index * batch_size, min((iter_index + 1) * batch_size, n_test)
        cost, _, acc = test_model(start, end, 0)
        cost_avg += cost * (end - start)
        acc_avg += acc * (end - start)
    cost_avg /= n_test
    acc_avg /= n_test
    costs_val.append(cost_avg)
    accs_val.append(acc_avg)
    print '\tTest cost = %f,\tAccuracy = %f' % (cost_avg, acc_avg)
    return cost_avg, acc_avg


# model name can be added "bi-" as prefix and "-only" as suffix
def train(X_train, y_train, X_test, y_test, model_name='bi-naive', hidden_dim=None, batch_size=32, num_epoch=15):

    n_train = X_train.shape[0]
    input_dim = X_train.shape[2]
    X_train = X_train.transpose([1, 0, 2])
    X_test = X_test.transpose([1, 0, 2])

    X_train_shared = theano.shared(X_train, borrow=True)
    y_train_shared = T.cast(theano.shared(y_train, borrow=True), 'int32')
    X_test_shared = theano.shared(X_test, borrow=True)
    y_test_shared = T.cast(theano.shared(y_test, borrow=True), 'int32')

    print model_name

    if 'bi' in model_name:
        model_name = model_name[3:]
        ra = Bi_RNN_Attention(input_dim, hidden_dim, 2, rnn=model_name, drop=0.2)
    else:
        ra = RNN_Attention(input_dim, hidden_dim, 2, rnn=model_name)
    symbols = ra.build_model()

    X_batch, y_batch, is_train = symbols['X_batch'], symbols['y_batch'], symbols['is_train']
    att, pred, acc = symbols['a'], symbols['pred'], symbols['acc']
    cost, updates = symbols['cost'], symbols['updates']

    num_iter = int(ceil(n_train / float(batch_size)))

    print 'Compiling function'

    start_symbol, end_symbol = T.lscalar(), T.lscalar()

    train_model = theano.function(inputs=[start_symbol, end_symbol, is_train],
                                  outputs=[cost, pred, acc], updates=updates,
                                  givens={
                                      X_batch: X_train_shared[:, start_symbol: end_symbol, :],
                                      y_batch: y_train_shared[start_symbol: end_symbol]},
                                  on_unused_input='ignore', mode='FAST_RUN')
    print 'Compilation done 1'
    test_model = theano.function(inputs=[start_symbol, end_symbol, is_train],
                                  outputs=[cost, pred, acc],
                                  givens={
                                      X_batch: X_test_shared[:, start_symbol: end_symbol, :],
                                      y_batch: y_test_shared[start_symbol: end_symbol]},
                                  on_unused_input='ignore', mode='FAST_RUN')
    print 'Compilation done 2'

    costs_train, costs_val = [], []
    accs_train, accs_val = [], []

    for epoch_index in xrange(num_epoch):
        cost_avg, acc_avg = 0., 0.
        print 'Epoch = %d' % (epoch_index + 1)
        for iter_index in xrange(num_iter):
            start, end = iter_index * batch_size, min((iter_index + 1) * batch_size, n_train)
            cost, _, acc = train_model(start, end, 1)
            cost_avg += cost * (end - start)
            acc_avg += acc * (end - start)
        cost_avg /= n_train
        acc_avg /= n_train
        costs_train.append(cost_avg)
        accs_train.append(acc_avg)
        print '\tTrain cost = %f,\tAccuracy = %f' % (cost_avg, acc_avg)
        validate(test_model, y_test, costs_val, accs_val)
    return costs_train, costs_val, accs_train, accs_val


def cross_validation(feature_name='hog', side='b'):

    feature_hidden = {'hog': 256, 'gemo': 128, 'au': 48, 'AU': 48, 'audio': 64}

    from data_preprocessing.load_data import load, load_audio
    from data_path import sample_10_root
    if feature_name == 'au' or feature_name == 'AU':
        tmp = 'gemo'
    else:
        tmp = feature_name
    # Use both speakers with adding features
    if feature_name == 'audio':
        dyad_features, dyad_ratings, dyad_slices = load_audio(side=side)
    else:
        dyad_features, dyad_ratings, dyad_slices = load(sample_10_root, feature_name=tmp, side=side)
    dyads = dyad_features.keys()
    hidden_dim = feature_hidden[feature_name]
    if feature_name == 'au' or feature_name == 'AU':
        for dyad, features in dyad_features.items():
            dyad_features[dyad] = features[:, :, -35:]

    # Create comparison features
    dyad_X, dyad_y = {}, {}
    for dyad, features in dyad_features.items():
        print dyad
        X = []
        y = []
        ratings = dyad_ratings[dyad]
        for i in xrange(features.shape[0]):
            for j in xrange(features.shape[0]):
                if i == j:
                    continue
                if abs(ratings[i] - ratings[j]) < 2.:
                    continue
                r = random.random()
                if r > 0.2:
                    continue
                if ratings[i] < ratings[j]:
                    X.append(features[i] - features[j])
                    y.append(0)
                else:
                    X.append(features[i] - features[j])
                    y.append(1)
        X = numpy.stack(X, axis=0).astype(theano.config.floatX)
        y = numpy.stack(y, axis=0).astype(theano.config.floatX)
        dyad_X[dyad] = X
        dyad_y[dyad] = y
        print X.shape, y.shape

    num_dyad = len(dyads)
    message = feature_name + '_' + side + '_comp'
    for i in xrange(num_dyad):
        dyad = dyads[i]
        X_test = dyad_X[dyad]
        y_test = dyad_y[dyad]
        feature_list, y_list = [], []
        for j in xrange(num_dyad):
            if j == i:
                continue
            feature_list.append(dyad_X[dyads[j]])
            y_list.append(dyad_y[dyads[j]])
        X_train = numpy.concatenate(feature_list)
        y_train = numpy.concatenate(y_list)
        print X_train.shape, X_test.shape
        print 'Testing Dyad =', dyad
        costs_train, costs_val, accs_train, accs_val = train(X_train, y_train, X_test, y_test, hidden_dim=hidden_dim)
        img_name = 'loss_dyad_' + str(dyad) + '_' + message + '.png'
        img_path = '../figs/' + img_name
        plot_loss(img_path, costs_train, costs_val, accs_train, accs_val)


def test1():
    parser = argparse.ArgumentParser()
    parser.add_argument('-feat', type=str, default='hog')
    parser.add_argument('-side', type=str, default=None)
    args = parser.parse_args()
    if args.side is not None:
        side = args.side
    else:
        if args.feat == 'audio' or args.feat == 'au' or args.feat == 'AU':
            side = 'b'
        else:
            side = 'ba'
    print args.feat, side
    cross_validation(feature_name=args.feat, side=side)


if __name__ == '__main__':
    test1()
