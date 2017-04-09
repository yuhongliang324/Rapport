__author__ = 'yuhongliang324'

from math import sqrt
import sys

import numpy
import theano
import theano.tensor as T
from sklearn.metrics import mean_squared_error

from models.rnn_attention import RNN_Attention

sys.path.append('../')
from models.rnn import RNN
from models.tagm import TAGM
from models.dan import dan
from utils import num_class


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


def validate(val_model, X_test, y_test, start_batches_test, end_batches_test, len_batches_test,
             category=False, need_attention=False):
    n_test = y_test.shape[0]
    num_iter = len(start_batches_test)
    all_pred = []
    all_attention = []
    cost_avg, loss_krip_avg = 0., 0.
    for iter_index in xrange(num_iter):
        start, end = start_batches_test[iter_index], end_batches_test[iter_index]
        length = len_batches_test[iter_index]
        xb = X_test[start: end, :length].T
        cost, tmp, pred, attention = val_model(xb, start, end, 0)
        if need_attention:
            all_attention.append(attention)
        cost_avg += cost * (end - start)
        if not category:
            loss_krip = tmp
            loss_krip_avg += loss_krip * (end - start)
        all_pred += pred.tolist()
    cost_avg /= n_test
    mae_acc = eval(y_test, all_pred, category=category)
    if category:
        print '\tTest cost = %f,\tAccuracy = %f' % (cost_avg, mae_acc)
    else:
        print '\tTest cost = %f,\tKrip Loss = %f,\tMAE = %f' % (cost_avg, loss_krip_avg, mae_acc)
    if not need_attention:
        all_attention = None
    return mae_acc, all_pred, all_attention


def train(E,
          X_train, y_train, start_batches_train, end_batches_train, len_batches_train,
          X_test, y_test, start_batches_test, end_batches_test, len_batches_test,
          drop=0.25, dec=True, update='adam',
          hidden_dim=None, num_epoch=60, lamb=0., model='ours', share=False, category=False,
          activation='tanh', need_attention=False, sq_loss=False):

    n_train = X_train.shape[0]
    input_dim = E.shape[-1]

    X_train_shared = theano.shared(X_train, borrow=True)
    y_train_shared = theano.shared(y_train, borrow=True)
    X_test_shared = theano.shared(X_test, borrow=True)
    y_test_shared = theano.shared(y_test, borrow=True)
    E_shared = theano.shared(E.astype(theano.config.floatX), borrow=True)

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

    num_iter = len(start_batches_train)

    print 'Compiling function'

    if category:
        outputs = [cost, acc, pred]
    else:
        outputs = [cost, loss_krip, pred]
    if model == 'ours':
        outputs.append(att)
    else:
        outputs.append(pred)  # trivial append

    start_symbol, end_symbol = T.lscalar(), T.lscalar()
    xb_symbol = T.imatrix()

    train_model = theano.function(inputs=[xb_symbol, start_symbol, end_symbol, is_train],
                                  outputs=outputs, updates=updates,
                                  givens={
                                      X_batch: E_shared[xb_symbol],
                                      y_batch: y_train_shared[start_symbol: end_symbol]},
                                  on_unused_input='ignore', mode='FAST_RUN')
    print 'Compilation done 1'
    test_model = theano.function(inputs=[xb_symbol, start_symbol, end_symbol, is_train],
                                 outputs=outputs,
                                 givens={
                                     X_batch: E_shared[xb_symbol],
                                     y_batch: y_test_shared[start_symbol: end_symbol]},
                                 on_unused_input='ignore', mode='FAST_RUN')
    print 'Compilation done 2'

    best_mae_acc = None
    best_pred_test = None
    best_epoch = 0
    best_att_test = None
    for epoch_index in xrange(num_epoch):
        cost_avg, loss_krip_avg, rmse = 0., 0., 0.
        all_pred = []
        print 'Epoch = %d' % (epoch_index + 1)
        for iter_index in xrange(num_iter):
            start, end = start_batches_train[iter_index], end_batches_train[iter_index]
            length = len_batches_train[iter_index]
            xb = X_train[start: end, :length].T
            print xb.shape, start, end
            cost, tmp, pred, attention = train_model(xb, start, end, 1)
            print cost, tmp, pred.shape, attention.shape
            cost_avg += cost * (end - start)
            if not category:
                loss_krip = tmp
                loss_krip_avg += loss_krip * (end - start)
            all_pred += pred.tolist()
        cost_avg /= n_train
        if not category:
            loss_krip_avg /= n_train
        y_predicted = numpy.asarray(all_pred)
        rmse_acc = eval(y_train, y_predicted, category=category)
        if category:
            print '\tTrain cost = %f,\tAccuracy = %f' % (cost_avg, rmse_acc)
        else:
            print '\tTrain cost = %f,\tKrip Loss = %f,\tRMSE = %f' % (cost_avg, loss_krip_avg, rmse_acc)
        mae_acc_test, pred_test, att_test\
            = validate(test_model, X_test, y_test, start_batches_test, end_batches_test, len_batches_test,
                       category=category, need_attention=need_attention)
        if (best_mae_acc is None) or (category and mae_acc_test > best_mae_acc)\
                or ((not category) and mae_acc_test < best_mae_acc):
            best_mae_acc = mae_acc_test
            best_pred_test = pred_test
            best_epoch = epoch_index
            best_att_test = att_test
        # Early Stopping
        if epoch_index - best_epoch >= 5 and epoch_index >= num_epoch // 4 and best_epoch > 2:
            break
    print 'Best Epoch = %d, Best MAE/ACC in Test = %f' % (best_epoch + 1, best_mae_acc)
    # Krip losses only make sense for regression (category = False)
    return best_pred_test, best_att_test
