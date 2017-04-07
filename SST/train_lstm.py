__author__ = 'yuhongliang324'

import sys

import theano

sys.path.append('../')
from models.lstm_old import LSTM
import argparse
from utils import load_data, train_pkl, valid_pkl, test_pkl, num_class


def train(drop=0., hidden_dim=None, lamb=0.0001, bidirection=False, update='adam2', batch_size=25, num_epoch=10):

    X_batches_train, y_batches_train = load_data(train_pkl, batch_size=batch_size)
    X_batches_val, y_batches_val = load_data(valid_pkl, batch_size=batch_size)
    X_batches_test, y_batches_test = load_data(test_pkl, batch_size=batch_size)

    input_dim = X_batches_train[0].shape[-1]

    X_batches_train = [Xb.transpose([1, 0, 2]) for Xb in X_batches_train]
    X_batches_val = [Xb.transpose([1, 0, 2]) for Xb in X_batches_val]
    X_batches_test = [Xb.transpose([1, 0, 2]) for Xb in X_batches_test]

    model = LSTM(input_dim, hidden_dim, [num_class], drop=drop, lamb=lamb, bidirection=bidirection, update=update)
    symbols = model.build_model()

    X_batch, y_batch, is_train = symbols['X_batch'], symbols['y_batch'], symbols['is_train']
    pred, loss, acc = symbols['pred'], symbols['loss'], symbols['acc']
    cost, updates = symbols['cost'], symbols['updates']

    print 'Compiling function'

    Xb_train = theano.shared(X_batches_train[0], borrow=True)
    yb_train = theano.shared(y_batches_train[0].astype('int32'), borrow=True)
    Xb_val = theano.shared(X_batches_val[0], borrow=True)
    yb_val = theano.shared(y_batches_val[0].astype('int32'), borrow=True)

    train_model = theano.function(inputs=[is_train],
                                  outputs=[cost, acc, pred], updates=updates,
                                  givens={
                                          X_batch: Xb_train,
                                          y_batch: yb_train},
                                  on_unused_input='ignore', mode='FAST_RUN')
    print 'Compilation done 1'
    valid_model = theano.function(inputs=[is_train],
                                  outputs=[cost, acc, pred],
                                  givens={
                                          X_batch: Xb_val,
                                          y_batch: yb_val},
                                  on_unused_input='ignore', mode='FAST_RUN')
    print 'Compilation done 2'

    num_batches_train = len(X_batches_train)
    num_batches_val = len(X_batches_val)
    num_batches_test = len(X_batches_test)
    for epoch_index in xrange(num_epoch):
        # Training
        cost_ep, acc_ep = 0., 0.
        total = 0.
        print 'Epoch = %d' % (epoch_index + 1)
        for iter_index in xrange(num_batches_train):
            Xb_train.set_value(X_batches_train[iter_index])
            yb_train.set_value(y_batches_train[iter_index].astype('int32'))
            cost, acc, pred = train_model(1)
            bs = X_batches_train[iter_index].shape[1]
            cost_ep += cost * bs
            acc_ep += acc * bs
            total += bs
        cost_ep /= total
        acc_ep /= total
        print '\tTrain cost = %f,\tAccuracy = %f' % (cost_ep, acc_ep)

        # Validation
        cost_ep, acc_ep = 0., 0.
        total = 0.
        for iter_index in xrange(num_batches_val):
            Xb_val.set_value(X_batches_val[iter_index])
            yb_val.set_value(y_batches_val[iter_index].astype('int32'))
            cost, acc, pred = valid_model(0)
            bs = X_batches_val[iter_index].shape[1]
            cost_ep += cost * bs
            acc_ep += acc * bs
            total += bs
        cost_ep /= total
        acc_ep /= total
        print '\tValidation cost = %f,\tAccuracy = %f' % (cost_ep, acc_ep)

        # Testing
        cost_test, acc_test = 0., 0.
        total = 0.
        for iter_index in xrange(num_batches_test):
            Xb_val.set_value(X_batches_test[iter_index])
            yb_val.set_value(y_batches_test[iter_index].astype('int32'))
            cost, acc, pred = valid_model(0)
            bs = X_batches_test[iter_index].shape[1]
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
    parser.add_argument('-bi', type=bool, default=True)
    parser.add_argument('-update', type=str, default='adam2')
    args = parser.parse_args()
    train(drop=args.drop, hidden_dim=args.hid, lamb=args.lamb, bidirection=args.bi,
          update=args.update, num_epoch=args.epoch)


if __name__ == '__main__':
    test1()
