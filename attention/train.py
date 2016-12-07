__author__ = 'yuhongliang324'

from math import ceil

import theano
import theano.tensor as T

from rnn_attention import RNN_Attention


def train(X_train, y_train, X_test, y_test, hidden_dim=512, batch_size=8, num_epoch=100):

    n_train = X_train.shape[0]
    input_dim = X_train.shape[2]
    X_train = X_train.transpose([1, 0, 2])
    X_test = X_test.transpose([1, 0, 2])

    X_train_shared = theano.shared(X_train, borrow=True)
    y_train_shared = theano.shared(y_train, borrow=True)
    X_test_shared = theano.shared(X_test, borrow=True)
    y_test_shared = theano.shared(y_test, borrow=True)

    ra = RNN_Attention(input_dim, hidden_dim, 1)
    symbols = ra.build_model()

    X_batch, y_batch = symbols['X_batch'], symbols['y_batch']
    att, pred, loss = symbols['a'], symbols['pred'], symbols['loss']
    cost, updates = symbols['cost'], symbols['updates']

    num_iter = int(ceil(n_train / float(batch_size)))

    print 'Compiling function'

    start_symbol, end_symbol = T.lscalar(), T.lscalar()

    train_model = theano.function(inputs=[start_symbol, end_symbol],
                                  outputs=[cost], updates=updates,
                                  givens={
                                      X_batch: X_train_shared[:, start_symbol: end_symbol, :],
                                      y_batch: y_train_shared[start_symbol: end_symbol]},
                                  on_unused_input='ignore')
    test_model = theano.function(inputs=[start_symbol, end_symbol],
                                  outputs=[cost], updates=updates,
                                  givens={
                                      X_batch: X_test_shared[:, start_symbol: end_symbol, :],
                                      y_batch: y_test_shared[start_symbol: end_symbol]},
                                  on_unused_input='ignore')
    print 'Compilation done'

    for epoch_index in xrange(num_epoch):
        for iter_index in xrange(num_iter):
            start, end = iter_index * batch_size, min((iter_index + 1) * batch_size, n_train)
            cost = train_model(start, end)
            cost = cost[0]
            print iter_index, cost


def test1():
    train()


if __name__ == '__main__':
    test1()
