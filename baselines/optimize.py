__author__ = 'yuhongliang324'

import numpy
import theano
import theano.tensor as T

from dan import dan
import sys
sys.path.append('../')

from attention.optimize import eval


def validate(val_model, y_val, costs_val, batch_size=32, category=False):
    n_test = y_val.shape[0]
    num_iter = (n_test + batch_size - 1) // batch_size
    all_pred = []
    cost_avg = 0.
    for iter_index in xrange(num_iter):
        start, end = iter_index * batch_size, min((iter_index + 1) * batch_size, n_test)
        cost, tmp, pred = val_model(start, end, 0)
        cost_avg += cost * (end - start)
        all_pred += pred.tolist()
    cost_avg /= n_test
    costs_val.append(cost_avg)
    rmse_acc = eval(y_val, all_pred, category=category)
    if category:
        print '\tTest cost = %f,\tAccuracy = %f' % (cost_avg, rmse_acc)
    else:
        print '\tTest cost = %f,\tRMSE = %f' % (cost_avg, rmse_acc)
    return cost_avg, all_pred


# model name can be added "-only" as suffix
def train(X_train, y_train, X_val, y_val, X_test, y_test, layers, drop=0.25, batch_size=64, num_epoch=100,
          category=False, activation='relu'):

    n_train = X_train.shape[0]
    X_train = X_train.transpose([1, 0, 2])
    X_val = X_val.transpose([1, 0, 2])
    X_test = X_test.transpose([1, 0, 2])

    X_train_shared = theano.shared(X_train, borrow=True)
    y_train_shared = theano.shared(y_train, borrow=True)
    X_val_shared = theano.shared(X_val, borrow=True)
    y_val_shared = theano.shared(y_val, borrow=True)
    X_test_shared = theano.shared(X_test, borrow=True)
    y_test_shared = theano.shared(y_test, borrow=True)

    model = dan(layers, update='adam', activation=activation, drop=drop)
    symbols = model.build_model()

    X_batch, y_batch, is_train = symbols['X_batch'], symbols['y_batch'], symbols['is_train']
    pred, loss = symbols['pred'], symbols['loss']
    cost, updates = symbols['cost'], symbols['updates']
    acc = symbols['acc']

    num_iter = (n_train + batch_size - 1) // batch_size

    print 'Compiling function'

    start_symbol, end_symbol = T.lscalar(), T.lscalar()
    if category:
        outputs = [cost, acc, pred]
    else:
        outputs = [cost, loss, pred]

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
    best_cost_val = 10000
    best_pred_test = None
    best_epoch = 0
    for epoch_index in xrange(num_epoch):
        cost_avg, rmse = 0., 0.
        all_pred = []
        print 'Epoch = %d' % (epoch_index + 1)
        for iter_index in xrange(num_iter):
            start, end = iter_index * batch_size, min((iter_index + 1) * batch_size, n_train)
            cost, tmp, pred = train_model(start, end, 1)
            cost_avg += cost * (end - start)
            all_pred += pred.tolist()
        cost_avg /= n_train
        costs_train.append(cost_avg)
        y_predicted = numpy.asarray(all_pred)
        rmse_acc = eval(y_train, y_predicted, category=category)
        if category:
            print '\tTrain cost = %f,\tAccuracy = %f' % (cost_avg, rmse_acc)
        else:
            print '\tTrain cost = %f,\tRMSE = %f' % (cost_avg, rmse_acc)
        cost_avg_val, _ = validate(valid_model, y_val, costs_val, category=category)
        _, pred_test = validate(test_model, y_test, costs_test, category=category)
        if cost_avg_val < best_cost_val:
            best_cost_val = cost_avg_val
            best_pred_test = pred_test
            best_epoch = epoch_index
        # Early Stopping
        if epoch_index - best_epoch >= 5 and epoch_index >= num_epoch // 2 and best_epoch > 2:
            return costs_train, costs_val, costs_test, best_pred_test
    return costs_train, costs_val, costs_test, best_pred_test
