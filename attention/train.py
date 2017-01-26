__author__ = 'yuhongliang324'

from math import ceil, sqrt
import numpy
import theano
import theano.tensor as T
from sklearn.metrics import mean_squared_error

from rnn_attention import RNN_Attention
from bi_rnn_attention import Bi_RNN_Attention
import sys
sys.path.append('../')
import time
from utils import plot_loss


def RMSE(y_actual, y_predicted):
    rmse = sqrt(mean_squared_error(y_actual, y_predicted))
    return rmse


def validate(test_model, y_test, costs_val, batch_size=32):
    n_test = y_test.shape[0]
    num_iter = int(ceil(n_test / float(batch_size)))
    all_pred = []
    cost_avg = 0.
    for iter_index in xrange(num_iter):
        start, end = iter_index * batch_size, min((iter_index + 1) * batch_size, n_test)
        cost, pred = test_model(start, end, 0)
        cost_avg += cost * (end - start)
        all_pred += pred.tolist()
    cost_avg /= n_test
    costs_val.append(cost_avg)
    rmse = RMSE(y_test, all_pred)
    print '\tTest cost = %f,\tRMSE = %f' % (cost_avg, rmse)
    return all_pred


# model name can be added "bi-" as prefix and "-only" as suffix
def train(X_train, y_train, X_test, y_test, model_name='bi-naive', hidden_dim=256, batch_size=32, num_epoch=30):

    n_train = X_train.shape[0]
    input_dim = X_train.shape[2]
    X_train = X_train.transpose([1, 0, 2])
    X_test = X_test.transpose([1, 0, 2])

    X_train_shared = theano.shared(X_train, borrow=True)
    y_train_shared = theano.shared(y_train, borrow=True)
    X_test_shared = theano.shared(X_test, borrow=True)
    y_test_shared = theano.shared(y_test, borrow=True)

    print model_name

    if 'bi' in model_name:
        model_name = model_name[3:]
        ra = Bi_RNN_Attention(input_dim, hidden_dim, 1, rnn=model_name, drop=0.)
    else:
        ra = RNN_Attention(input_dim, hidden_dim, 1, rnn=model_name)
    symbols = ra.build_model()

    X_batch, y_batch, is_train = symbols['X_batch'], symbols['y_batch'], symbols['is_train']
    att, pred, loss = symbols['a'], symbols['pred'], symbols['loss']
    cost, updates = symbols['cost'], symbols['updates']

    num_iter = int(ceil(n_train / float(batch_size)))

    print 'Compiling function'

    start_symbol, end_symbol = T.lscalar(), T.lscalar()

    train_model = theano.function(inputs=[start_symbol, end_symbol, is_train],
                                  outputs=[cost, pred], updates=updates,
                                  givens={
                                      X_batch: X_train_shared[:, start_symbol: end_symbol, :],
                                      y_batch: y_train_shared[start_symbol: end_symbol]},
                                  on_unused_input='ignore', mode='FAST_RUN')
    print 'Compilation done 1'
    test_model = theano.function(inputs=[start_symbol, end_symbol, is_train],
                                  outputs=[cost, pred],
                                  givens={
                                      X_batch: X_test_shared[:, start_symbol: end_symbol, :],
                                      y_batch: y_test_shared[start_symbol: end_symbol]},
                                  on_unused_input='ignore', mode='FAST_RUN')
    print 'Compilation done 2'

    costs_train, costs_val = [], []

    for epoch_index in xrange(num_epoch):
        cost_avg, rmse = 0., 0.
        all_pred = []
        print 'Epoch = %d' % (epoch_index + 1)
        for iter_index in xrange(num_iter):
            start, end = iter_index * batch_size, min((iter_index + 1) * batch_size, n_train)
            cost, pred = train_model(start, end, 1)
            cost_avg += cost * (end - start)
            all_pred += pred.tolist()
        cost_avg /= n_train
        costs_train.append(cost_avg)
        y_predicted = numpy.asarray(all_pred)
        rmse = RMSE(y_train, y_predicted)
        print '\tTrain cost = %f,\tRMSE = %f' % (cost_avg, rmse)
        pred_val = validate(test_model, y_test, costs_val)
    return costs_train, costs_val, pred_val


def cross_validation(feature_name='hog'):

    feature_hidden = {'hog': 256, 'gemo': 128, 'au': 48, 'AU': 48, 'audio': 64}

    from data_preprocessing.load_data import load, load_audio
    from data_path import sample_10_root
    if feature_name == 'au' or feature_name == 'AU':
        tmp = 'gemo'
    else:
        tmp = feature_name
    # Use both speakers with adding features
    if feature_name == 'audio':
        dyad_features, dyad_ratings, _ = load_audio()
    else:
        dyad_features, dyad_ratings = load(sample_10_root, feature_name=tmp, side='ba')
    dyads = dyad_features.keys()
    hidden_dim = feature_hidden[feature_name]
    if feature_name == 'au' or feature_name == 'AU':
        for dyad, features in dyad_features.items():
            dyad_features[dyad] = features[:, :, -35:]
    num_dyad = len(dyads)
    writer = open('../results/result.txt', 'w')
    for i in xrange(num_dyad):
        dyad = dyads[i]
        X_test = dyad_features[dyad]
        y_test = dyad_ratings[dyad]
        feature_list, rating_list = [], []
        for j in xrange(num_dyad):
            if j == i:
                continue
            feature_list.append(dyad_features[dyads[j]])
            rating_list.append(dyad_ratings[dyads[j]])
        X_train = numpy.concatenate(feature_list)
        y_train = numpy.concatenate(rating_list)
        rating_mean = numpy.mean(y_train)
        rmse = y_test - rating_mean
        rmse = sqrt(numpy.mean(rmse * rmse))
        print 'Testing Dyad =', dyad
        print 'RMSE of Average Prediction = %f' % rmse
        print X_train.shape, X_test.shape
        costs_train, costs_val, pred_val = train(X_train, y_train, X_test, y_test, hidden_dim=hidden_dim)
        img_name = 'loss_dyad_' + str(dyad) + '_' + time.strftime('%m%d-%H%M%S', time.localtime()) + '.png'
        img_path = '../figs/' + img_name
        plot_loss(img_path, costs_train, costs_val)
        for i in xrange(y_test.shape[0]):
            writer.write(str(dyad) + ',' + str(y_test[i]) + ',' + str(pred_val[i]) + '\n')


def test1():
    cross_validation(feature_name='audio')


if __name__ == '__main__':
    test1()
