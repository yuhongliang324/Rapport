__author__ = 'yuhongliang324'

import theano
import theano.tensor as T
from load_data import load, n_class
from rnn_attention import RNN_Attention
from math import ceil


def train(batch_size=8, num_epoch=100):

    X, y = load()
    X = theano.shared(X, borrow=True)
    y = theano.shared(y, borrow=True)
    y = T.cast(y, 'int32')

    ra = RNN_Attention(1000, 500, n_class)
    symbols = ra.build_model()

    X_batch, y_batch = symbols['X_batch'], symbols['y_batch']
    cost, updates = symbols['cost'], symbols['updates']

    n_train = X.get_values().shape[0]
    num_iter = int(ceil(n_train / float(batch_size)))

    print 'Compiling function'

    start_symbol, end_symbol = T.lscalar(), T.lscalar()

    train_model = theano.function(inputs=[start_symbol, end_symbol],
                                  outputs=[cost], updates=updates,
                                  givens={
                                      X_batch: X[:, start_symbol: end_symbol, :],
                                      y_batch: y[start_symbol: end_symbol],
                                  },
                                  on_unused_input='ignore')
    print 'Compilation done'

    for epoch_index in xrange(num_epoch):
        for iter_index in xrange(num_iter):
            start, end = iter_index * batch_size, min((iter_index + 1) * batch_size, n_train)
            cost = train_model(start, end)
            print iter_index, cost


def test1():
    train()


if __name__ == '__main__':
    test1()
