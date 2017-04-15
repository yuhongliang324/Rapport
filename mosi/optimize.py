__author__ = 'yuhongliang324'

from math import sqrt
import sys

import numpy
import theano
import theano.tensor as T
from sklearn.metrics import mean_squared_error

sys.path.append('../')

from models.rnn_attention import RNN_Attention
from models.dan import dan
from models.tagm import TAGM
from models.rnn import RNN


def eval(y_actual, y_predicted, category=False):
    if category:
        right = 0.
        for i in xrange(y_actual.shape[0]):
            if y_actual[i] == y_predicted[i]:
                right += 1
        acc = float(right) / y_actual.shape[0]
        return acc
    else:
        return MAE(y_actual, y_predicted)


def RMSE(y_actual, y_predicted):
    rmse = sqrt(mean_squared_error(y_actual, y_predicted))
    return rmse


def MAE(y_actual, y_predicted):
    dist = 0.
    for i in xrange(y_actual.shape[0]):
        dist += abs(y_actual[i] - y_predicted[i])
    mae = dist / y_actual.shape[0]
    return mae


def test(test_model, start_batches_test, end_batches_test, len_batches_test,
         y_test, costs_test, category=False):
    n_test = y_test.shape[0]
    all_actual, all_pred = [], []
    cost_avg = 0.
    num_iter = len(start_batches_test)
    for iter_index in xrange(num_iter):
        start, end = start_batches_test[iter_index], end_batches_test[iter_index]
        length = len_batches_test[iter_index]
        if length == 0:
            # all_actual += y_test[start: end].tolist()
            # all_pred += [-100] * (end - start)
            continue
        cost, tmp, pred = test_model(start, end, length, 0)
        cost_avg += cost * (end - start)
        all_actual += y_test[start: end].tolist()
        all_pred += pred.tolist()
        if (iter_index + 1) % 100 == 0:
                print iter_index + 1, '/', num_iter
    cost_avg /= n_test
    costs_test.append(cost_avg)
    y_actual = numpy.asarray(all_actual)
    y_predicted = numpy.asarray(all_pred)
    mae_acc = eval(y_actual, y_predicted, category=category)
    if category:
        print '\tTest cost = %f,\tAccuracy = %f' % (cost_avg, mae_acc)
    else:
        print '\tTest cost = %f,\tMAE = %f' % (cost_avg, mae_acc)
    return mae_acc, y_actual, y_predicted


def train(inputs_train, inputs_test, hidden_dim=None, dec=True, update='adam', sq_loss=False, activation=None,
          lamb=0., drop=0., model='gru', share=False, category=False, num_epoch=40, num_class=None):

    Xs_train, y_train, start_batches_train, end_batches_train, len_batches_train = inputs_train
    Xs_test, y_test, start_batches_test, end_batches_test, len_batches_test = inputs_test

    n_train = Xs_train.shape[0]
    input_dim = Xs_train.shape[2]
    Xs_train = Xs_train.transpose([1, 0, 2])
    Xs_test = Xs_test.transpose([1, 0, 2])

    Xs_train_shared = theano.shared(Xs_train, borrow=True)
    y_train_shared = theano.shared(y_train, borrow=True)
    Xs_test_shared = theano.shared(Xs_test, borrow=True)
    y_test_shared = theano.shared(y_test, borrow=True)

    if category:
        n_class = num_class
    else:
        n_class = 1

    if model == 'ours':
        ra = RNN_Attention(input_dim, hidden_dim, [n_class], dec=dec, drop=drop,
                           update=update, lamb=lamb, model='lstm', share=share, sq_loss=sq_loss)
    elif model == 'tagm':
        ra = TAGM(input_dim, hidden_dim, [n_class], lamb=lamb, update=update, drop=drop, activation=activation,
                  sq_loss=sq_loss)
    elif model == 'dan':
        ra = dan([input_dim, min(512, int(0.5 * input_dim)), min(256, int(0.5 * input_dim)), n_class],
                 lamb=lamb, update=update, activation=activation, drop=drop, sq_loss=sq_loss)
    else:
        ra = RNN(input_dim, hidden_dim, [n_class], lamb=lamb, model=model, share=share, update=update, drop=drop,
                 sq_loss=sq_loss)
    symbols = ra.build_model()

    X_batch, y_batch, is_train = symbols['X_batch'], symbols['y_batch'], symbols['is_train']
    att, pred, loss = symbols['att'], symbols['pred'], symbols['loss']
    cost, updates = symbols['cost'], symbols['updates']
    loss_krip, acc = symbols['loss_krip'], symbols['acc']

    print 'Compiling function'

    start_symbol, end_symbol = T.lscalar(), T.lscalar()
    len_symbol = T.iscalar()
    if category:
        outputs = [cost, acc, pred]
    else:
        outputs = [cost, loss_krip, pred]

    train_model = theano.function(inputs=[start_symbol, end_symbol, len_symbol, is_train],
                                  outputs=outputs, updates=updates,
                                  givens={
                                      X_batch: Xs_train_shared[:len_symbol, start_symbol: end_symbol, :],
                                      y_batch: y_train_shared[start_symbol: end_symbol]},
                                  on_unused_input='ignore', mode='FAST_RUN')
    print 'Compilation done 1'
    test_model = theano.function(inputs=[start_symbol, end_symbol, len_symbol, is_train],
                                 outputs=outputs,
                                 givens={
                                     X_batch: Xs_test_shared[:len_symbol, start_symbol: end_symbol, :],
                                     y_batch: y_test_shared[start_symbol: end_symbol]},
                                 on_unused_input='ignore', mode='FAST_RUN')
    print 'Compilation done 2'

    costs_train, costs_test = [], []
    best_mae_acc = None
    best_actual_test = None
    best_pred_test = None
    best_epoch = 0
    num_iter = len(start_batches_train)
    for epoch_index in xrange(num_epoch):
        cost_avg = 0.
        all_actual, all_pred = [], []
        print 'Epoch = %d' % (epoch_index + 1)
        for iter_index in xrange(num_iter):
            start, end = start_batches_train[iter_index], end_batches_train[iter_index]
            length = len_batches_train[iter_index]
            if length == 0:
                continue
            cost, tmp, pred = train_model(start, end, length, 1)
            cost_avg += cost * (end - start)
            all_actual += y_train[start: end].tolist()
            all_pred += pred.tolist()
            if (iter_index + 1) % 100 == 0:
                print iter_index + 1, '/', num_iter
        cost_avg /= n_train
        costs_train.append(cost_avg)
        y_actual = numpy.asarray(all_actual)
        y_predicted = numpy.asarray(all_pred)

        mae_acc = eval(y_actual, y_predicted, category=category)
        if category:
            print '\tTrain cost = %f,\tAccuracy = %f' % (cost_avg, mae_acc)
        else:
            print '\tTrain cost = %f,\tMAE = %f' % (cost_avg, mae_acc)
        mae_acc_test, actual_test, pred_test = test(test_model, start_batches_test, end_batches_test, len_batches_test,
                                                    y_test, costs_test, category=category)
        if best_mae_acc is None:
            best_mae_acc = mae_acc_test
            best_actual_test = actual_test
            best_pred_test = pred_test
            best_epoch = epoch_index
        elif (category and mae_acc_test > best_mae_acc) or ((not category) and mae_acc_test < best_mae_acc):
            best_mae_acc = mae_acc_test
            best_actual_test = actual_test
            best_pred_test = pred_test
            best_epoch = epoch_index
        '''
        # Early Stopping
        if epoch_index - best_epoch >= 4 and epoch_index >= num_epoch // 4:
            break'''
    print 'Best Epoch = %d, Best ACC/MAE in Test = %f' % (best_epoch + 1, best_mae_acc)
    return best_actual_test, best_pred_test
