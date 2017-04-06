__author__ = 'yuhongliang324'

from math import sqrt
import numpy
import theano
import theano.tensor as T
from sklearn.metrics import mean_squared_error

from rnn_attention import RNN_Attention
import sys
sys.path.append('../')
from SST.lstm import LSTM
from tagm import TAGM


def eval(y_actual, y_predicted, category=False):
    if category:
        right = 0.
        for i in xrange(y_actual.shape[0]):
            if y_actual[i] == y_predicted[i]:
                right += 1
        acc = float(right) / y_actual.shape[0]
        return acc
    else:
        return RMSE(y_actual, y_predicted)


def RMSE(y_actual, y_predicted):
    rmse = sqrt(mean_squared_error(y_actual, y_predicted))
    return rmse


def validate(val_model, y_val, costs_val, losses_krip_val, batch_size=32, category=False):
    n_test = y_val.shape[0]
    num_iter = (n_test + batch_size - 1) // batch_size
    all_pred = []
    cost_avg, loss_krip_avg = 0., 0.
    for iter_index in xrange(num_iter):
        start, end = iter_index * batch_size, min((iter_index + 1) * batch_size, n_test)
        cost, tmp, pred = val_model(start, end, 0)
        cost_avg += cost * (end - start)
        if not category:
            loss_krip = tmp
            loss_krip_avg += loss_krip * (end - start)
        all_pred += pred.tolist()
    cost_avg /= n_test
    costs_val.append(cost_avg)
    if not category:
        loss_krip_avg /= n_test
        losses_krip_val.append(loss_krip_avg)
    rmse_acc = eval(y_val, all_pred, category=category)
    if category:
        print '\tTest cost = %f,\tAccuracy = %f' % (cost_avg, rmse_acc)
    else:
        print '\tTest cost = %f,\tKrip Loss = %f,\tRMSE = %f' % (cost_avg, loss_krip_avg, rmse_acc)
    return cost_avg, all_pred


# model name can be added "-only" as suffix
def train(X_train, y_train, X_val, y_val, X_test, y_test, drop=0.25, final_activation=None, dec=True, update='adam',
          hidden_dim=None, batch_size=64, num_epoch=60, lamb=0., model='ours', share=False, category=False):

    n_train = X_train.shape[0]
    input_dim = X_train.shape[2]
    X_train = X_train.transpose([1, 0, 2])
    X_val = X_val.transpose([1, 0, 2])
    X_test = X_test.transpose([1, 0, 2])

    X_train_shared = theano.shared(X_train, borrow=True)
    y_train_shared = theano.shared(y_train, borrow=True)
    X_val_shared = theano.shared(X_val, borrow=True)
    y_val_shared = theano.shared(y_val, borrow=True)
    X_test_shared = theano.shared(X_test, borrow=True)
    y_test_shared = theano.shared(y_test, borrow=True)

    if category:
        n_class = 3
    else:
        n_class = 1

    if model == 'ours':
        ra = RNN_Attention(input_dim, hidden_dim, [n_class], dec=dec, drop=drop, final_activation=final_activation,
                           update=update, lamb=lamb, model=model, share=share)
    elif model == 'tagm':
        ra = TAGM(input_dim, hidden_dim, [n_class], lamb=lamb, update=update, drop=drop)
    else:
        if model.startswith('s'):
            ra = LSTM(input_dim, hidden_dim, [n_class], lamb=lamb, update='adam2', drop=drop, bidirection=False)
        else:
            ra = LSTM(input_dim, hidden_dim, [n_class], lamb=lamb, update='adam2', drop=drop, bidirection=True)
    symbols = ra.build_model()

    X_batch, y_batch, is_train = symbols['X_batch'], symbols['y_batch'], symbols['is_train']
    att, pred, loss = symbols['att'], symbols['pred'], symbols['loss']
    cost, updates = symbols['cost'], symbols['updates']
    loss_krip, acc = symbols['loss_krip'], symbols['acc']

    num_iter = (n_train + batch_size - 1) // batch_size

    print 'Compiling function'

    start_symbol, end_symbol = T.lscalar(), T.lscalar()
    if category:
        outputs = [cost, acc, pred]
    else:
        outputs = [cost, loss_krip, pred]

    train_model = theano.function(inputs=[start_symbol, end_symbol, is_train],
                                  outputs=outputs, updates=updates,
                                  givens={
                                      X_batch: X_train_shared[:, start_symbol: end_symbol, :],
                                      y_batch: y_train_shared[start_symbol: end_symbol]},
                                  on_unused_input='ignore', mode='FAST_RUN')
    print 'Compilation done 1'
    valid_model = theano.function(inputs=[start_symbol, end_symbol, is_train],
                                  outputs=outputs,
                                  givens={
                                      X_batch: X_val_shared[:, start_symbol: end_symbol, :],
                                      y_batch: y_val_shared[start_symbol: end_symbol]},
                                  on_unused_input='ignore', mode='FAST_RUN')
    print 'Compilation done 2'
    test_model = theano.function(inputs=[start_symbol, end_symbol, is_train],
                                 outputs=outputs,
                                 givens={
                                     X_batch: X_test_shared[:, start_symbol: end_symbol, :],
                                     y_batch: y_test_shared[start_symbol: end_symbol]},
                                 on_unused_input='ignore', mode='FAST_RUN')
    print 'Compilation done 3'

    costs_train, costs_val, costs_test = [], [], []
    losses_krip_train, losses_krip_val, losses_krip_test = [], [], []
    best_cost_val = 10000
    best_pred_test = None
    best_epoch = 0
    for epoch_index in xrange(num_epoch):
        cost_avg, loss_krip_avg, rmse = 0., 0., 0.
        all_pred = []
        print 'Epoch = %d' % (epoch_index + 1)
        for iter_index in xrange(num_iter):
            start, end = iter_index * batch_size, min((iter_index + 1) * batch_size, n_train)
            cost, tmp, pred = train_model(start, end, 1)
            cost_avg += cost * (end - start)
            if not category:
                loss_krip = tmp
                loss_krip_avg += loss_krip * (end - start)
            all_pred += pred.tolist()
        cost_avg /= n_train
        costs_train.append(cost_avg)
        if not category:
            loss_krip_avg /= n_train
            losses_krip_train.append(loss_krip_avg)
        y_predicted = numpy.asarray(all_pred)
        rmse_acc = eval(y_train, y_predicted, category=category)
        if category:
            print '\tTrain cost = %f,\tAccuracy = %f' % (cost_avg, rmse_acc)
        else:
            print '\tTrain cost = %f,\tKrip Loss = %f,\tRMSE = %f' % (cost_avg, loss_krip_avg, rmse_acc)
        cost_avg_val, _ = validate(valid_model, y_val, costs_val, losses_krip_val, category=category)
        _, pred_test = validate(test_model, y_test, costs_test, losses_krip_test, category=category)
        if cost_avg_val < best_cost_val:
            best_cost_val = cost_avg_val
            best_pred_test = pred_test
            best_epoch = epoch_index
        # Early Stopping
        if epoch_index - best_epoch >= 5 and epoch_index >= num_epoch // 4 and best_epoch > 2:
            return costs_train, costs_val, costs_test,\
                   losses_krip_train, losses_krip_val, losses_krip_test, best_pred_test
    # Krip losses only make sense for regression (category = False)
    return costs_train, costs_val, costs_test, losses_krip_train, losses_krip_val, losses_krip_test, best_pred_test
