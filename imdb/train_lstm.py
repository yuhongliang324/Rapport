__author__ = 'yuhongliang324'

import theano
from theano import tensor as T
import sys
sys.path.append('../')
from lstm import LSTM
import argparse
from utils import load_data, load_dict, train_pkl, test_pkl, num_class


def train(drop=0., hidden_dim=None, lamb=0.0001, bidirection=False, update='adam2', batch_size=25, num_epoch=10):

    X_train, y_train, start_batches_train, end_batches_train, len_batches_train\
        = load_data(train_pkl, batch_size=batch_size)
    X_test, y_test, start_batches_test, end_batches_test, len_batches_test\
        = load_data(test_pkl, batch_size=batch_size)
    _, E = load_dict()
    ID_train = T.cast(theano.shared(X_train, borrow=True), 'int32')
    y_train = theano.shared(y_train, borrow=True)
    ID_test = T.cast(theano.shared(X_test, borrow=True), 'int32')
    y_test = theano.shared(y_test, borrow=True)

    E_shared = theano.shared(E.astype(theano.config.floatX), borrow=True)

    input_dim = E.shape[-1]

    model = LSTM(input_dim, hidden_dim, [num_class], drop=drop, lamb=lamb, bidirection=bidirection, update=update)
    symbols = model.build_model()

    E_sym, n_step, ID_batch = symbols['E'], symbols['n_step'], symbols['ID_batch']
    y_batch, is_train = symbols['y_batch'], symbols['is_train']
    pred, loss, acc = symbols['pred'], symbols['loss'], symbols['acc']
    cost, updates = symbols['cost'], symbols['updates']

    print 'Compiling function'

    start_symbol, end_symbol = T.lscalar(), T.lscalar()
    train_model = theano.function(inputs=[start_symbol, end_symbol, n_step, is_train],
                                  outputs=[cost, acc, pred], updates=updates,
                                  givens={E_sym: E_shared,
                                          ID_batch: ID_train[start_symbol: end_symbol],
                                          y_batch: y_train[start_symbol: end_symbol]},
                                  on_unused_input='ignore', mode='FAST_RUN')
    print 'Compilation done 1'
    test_model = theano.function(inputs=[start_symbol, end_symbol, n_step, is_train],
                                 outputs=[cost, acc, pred],
                                 givens={E_sym: E_shared,
                                         ID_batch: ID_test[start_symbol: end_symbol],
                                         y_batch: y_test[start_symbol: end_symbol]},
                                  on_unused_input='ignore', mode='FAST_RUN')
    print 'Compilation done 2'

    num_batches_train = len(start_batches_train)
    num_batches_test = len(start_batches_test)
    for epoch_index in xrange(num_epoch):
        # Training
        cost_ep, acc_ep = 0., 0.
        total = 0.
        print 'Epoch = %d' % (epoch_index + 1)
        for iter_index in xrange(num_batches_train):
            start, end = start_batches_train[iter_index], end_batches_train[iter_index]
            length = len_batches_train[iter_index]
            cost, acc, pred = train_model(start, end, length, 1)
            print '\titer = %d / %d, Cost = %f, Acc = %f' % (iter_index + 1, num_batches_train, cost, acc)
            bs = end - start
            cost_ep += cost * bs
            acc_ep += acc * bs
            total += bs
        cost_ep /= total
        acc_ep /= total
        print '\tTrain cost = %f,\tAccuracy = %f' % (cost_ep, acc_ep)

        # Testing
        cost_test, acc_test = 0., 0.
        total = 0.
        for iter_index in xrange(num_batches_test):
            start, end = start_batches_test[iter_index], end_batches_test[iter_index]
            length = len_batches_test[iter_index]
            cost, acc, pred = test_model(start, end, length, 0)
            bs = end - start
            cost_test += cost * bs
            acc_test += acc * bs
            total += bs
        cost_test /= total
        acc_test /= total
        print '\tTest cost = %f,\tAccuracy = %f' % (cost_test, acc_test)


def test1():
    parser = argparse.ArgumentParser()
    parser.add_argument('-hid', type=int, default=168)
    parser.add_argument('-drop', type=float, default=0.)
    parser.add_argument('-epoch', type=int, default=10)
    parser.add_argument('-lamb', type=float, default=0.0001)
    parser.add_argument('-bi', type=bool, default=False)
    parser.add_argument('-update', type=str, default='adam2')
    args = parser.parse_args()
    train(drop=args.drop, hidden_dim=args.hid, lamb=args.lamb, bidirection=args.bi,
          update=args.update, num_epoch=args.epoch)


if __name__ == '__main__':
    test1()
