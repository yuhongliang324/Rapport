__author__ = 'yuhongliang324'

from math import sqrt
import numpy
import theano
import theano.tensor as T
from sklearn.metrics import mean_squared_error
import sys
sys.path.append('../')

from attention.rnn_attention import RNN_Attention


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


def test(test_model, start_batches_test, end_batches_test, len_batches_test,
         y_test, costs_test, category=False):
    n_test = y_test.shape[0]
    all_pred = []
    cost_avg = 0.
    num_iter = len(start_batches_test)
    for iter_index in xrange(num_iter):
        start, end = start_batches_test[iter_index], end_batches_test[iter_index]
        length = len_batches_test[iter_index]
        cost, tmp, pred = test_model(start, end, length, 0)
        cost_avg += cost * (end - start)
        all_pred += pred.tolist()
        if (iter_index + 1) % 100 == 0:
                print iter_index + 1, '/', num_iter
    cost_avg /= n_test
    costs_test.append(cost_avg)
    rmse_acc = eval(y_test, all_pred, category=category)
    if category:
        print '\tTest cost = %f,\tAccuracy = %f' % (cost_avg, rmse_acc)
    else:
        print '\tTest cost = %f,\tRMSE = %f' % (cost_avg, rmse_acc)
    return cost_avg, all_pred


def train(inputs_train, inputs_test, hidden_dim=None, dec=True, update='adam',
          lamb=0., model='gru', share=False, category=False, num_epoch=50):

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
        n_class = 4
    else:
        n_class = 1

    ra = RNN_Attention(input_dim, hidden_dim, [n_class], dec=dec, update=update, lamb=lamb, model=model, share=share)
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
    best_cost_test = 10000
    best_pred_test = None
    best_epoch = 0
    num_iter = len(start_batches_train)
    for epoch_index in xrange(num_epoch):
        cost_avg, rmse = 0., 0.
        all_pred = []
        print 'Epoch = %d' % (epoch_index + 1)
        for iter_index in xrange(num_iter):
            start, end = start_batches_train[iter_index], end_batches_train[iter_index]
            length = len_batches_train[iter_index]
            cost, tmp, pred = train_model(start, end, length, 1)
            print pred.shape
            cost_avg += cost * (end - start)
            all_pred += pred.tolist()
            if (iter_index + 1) % 100 == 0:
                print iter_index + 1, '/', num_iter
        cost_avg /= n_train
        costs_train.append(cost_avg)
        y_predicted = numpy.asarray(all_pred)
        rmse_acc = eval(y_train, y_predicted, category=category)
        if category:
            print '\tTrain cost = %f,\tAccuracy = %f' % (cost_avg, rmse_acc)
        else:
            print '\tTrain cost = %f,\tRMSE = %f' % (cost_avg, rmse_acc)
        cost_avg_test, pred_test = test(test_model, start_batches_test, end_batches_test, len_batches_test,
                                        y_test, costs_test, category=category)
        if cost_avg_test < best_cost_test:
            best_cost_test = cost_avg_test
            best_pred_test = pred_test
            best_epoch = epoch_index
        # Early Stopping
        if epoch_index - best_epoch >= 5 and epoch_index >= num_epoch // 4 and best_epoch > 2:
            return best_pred_test
    # Krip losses only make sense for regression (category = False)
    return best_pred_test
