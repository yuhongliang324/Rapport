__author__ = 'yuhongliang324'

# The training and testing are comparisons

from math import ceil, sqrt
import numpy
import theano
import theano.tensor as T

from comp_net import ComparisonNet
import sys
sys.path.append('../')


class Model_Compiler:
    def __init__(self, input_dim, hidden_dim, n_class=1, fusion='conc'):
        cn = ComparisonNet(input_dim, hidden_dim, n_class=n_class, fusion=fusion)
        symbols = cn.build_model()
        self.X1_batch, self.X2_batch, self.y_batch = symbols['X1_batch'], symbols['X2_batch'], symbols['y_batch']
        self.cost, self.acc, self.updates = symbols['cost'], symbols['acc'], symbols['updates']
        self.pred = symbols['pred']
        self.train_model, self.test_model = None, None

    def compile(self, X1_train, X2_train, y_train, X1_test, X2_test, y_test):
        X1_train = X1_train.transpose([1, 0, 2])
        X2_train = X2_train.transpose([1, 0, 2])
        X1_test = X1_test.transpose([1, 0, 2])
        X2_test = X2_test.transpose([1, 0, 2])
        print 'Compiling function'
        if self.train_model is None:
            self.X1_train_shared = theano.shared(X1_train, borrow=True)
            self.X2_train_shared = theano.shared(X2_train, borrow=True)
            self.y_train_shared = T.cast(theano.shared(y_train, borrow=True), 'int32')
            self.X1_test_shared = theano.shared(X1_test, borrow=True)
            self.X2_test_shared = theano.shared(X2_test, borrow=True)
            self.y_test_shared = T.cast(theano.shared(y_test, borrow=True), 'int32')
            start_symbol, end_symbol = T.lscalar(), T.lscalar()
            self.train_model = theano.function(inputs=[start_symbol, end_symbol],
                                               outputs=[self.cost, self.acc], updates=self.updates,
                                               givens={
                                                   self.X1_batch: self.X1_train_shared[:, start_symbol: end_symbol, :],
                                                   self.X2_batch: self.X2_train_shared[:, start_symbol: end_symbol, :],
                                                   self.y_batch: self.y_train_shared[start_symbol: end_symbol]},
                                               on_unused_input='ignore')
            self.test_model = theano.function(inputs=[start_symbol, end_symbol],
                                              outputs=[self.cost, self.acc, self.pred],
                                              givens={
                                                  self.X1_batch: self.X1_test_shared[:, start_symbol: end_symbol, :],
                                                  self.X2_batch: self.X2_test_shared[:, start_symbol: end_symbol, :],
                                                  self.y_batch: self.y_test_shared[start_symbol: end_symbol]},
                                              on_unused_input='ignore')
        else:
            self.X1_train_shared.set_value(X1_train)
            self.X2_train_shared.set_value(X2_train)
            self.y_train_shared.set_value(y_train)
            self.X1_test_shared.set_value(X1_test)
            self.X2_test_shared.set_value(X2_test)
            self.y_test_shared.set_value(y_test)
        print 'Compiling function'


def validate(test_model, n_test, batch_size=32):
    num_iter = int(ceil(n_test / float(batch_size)))
    cost, acc = 0, 0
    for iter_index in xrange(num_iter):
        start, end = iter_index * batch_size, min((iter_index + 1) * batch_size, n_test)
        cost_iter, acc_iter, pred_iter = test_model(start, end)
        cost += cost_iter * (end - start)
        acc += acc_iter * (end - start)
    cost /= n_test
    acc /= n_test
    print '\tTest cost = %f,\tAccuracy = %f' % (cost, acc)


def optimize(train_model, test_model, n_train, n_test, batch_size=64, num_epoch=5):

    num_iter = int(ceil(n_train / float(batch_size)))

    for epoch_index in xrange(num_epoch):
        cost, acc = 0., 0.
        print 'Epoch = %d' % (epoch_index + 1)
        for iter_index in xrange(num_iter):
            start, end = iter_index * batch_size, min((iter_index + 1) * batch_size, n_train)
            cost_iter, acc_iter = train_model(start, end)
            cost += cost_iter * (end - start)
            acc += acc_iter * (end - start)
        cost /= n_train
        acc /= n_train
        print '\tTrain cost = %f,\tAccuracy = %f' % (cost, acc)
        validate(test_model, n_test)


def cross_validation(n_class, hidden_dim=128, fusion='conc'):
    from data_preprocessing.load_data import load_pairs
    from data_path import sample_10_root
    print 'Preparing pairs ... '
    dyad_X1, dyad_X2, dyad_gaps = load_pairs(sample_10_root, n_class=n_class)
    dyads = dyad_X1.keys()
    num_dyad = len(dyads)

    compiler = None

    for i in xrange(num_dyad):  # num_dyad
        dyad = dyads[i]
        X1_test = dyad_X1[dyad]
        X2_test = dyad_X2[dyad]
        y_test = dyad_gaps[dyad]
        X1_list, X2_list, rating_list = [], [], []
        for j in xrange(num_dyad):
            if j == i:
                continue
            X1_list.append(dyad_X1[dyads[j]])
            X2_list.append(dyad_X2[dyads[j]])
            rating_list.append(dyad_gaps[dyads[j]])
        X1_train = numpy.concatenate(X1_list)
        X2_train = numpy.concatenate(X2_list)
        gap_train = numpy.concatenate(rating_list)

        # shuffle
        print 'Shuffling ... '
        indices = numpy.arange(X1_train.shape[0])
        numpy.random.shuffle(indices)
        X1_train = X1_train[indices]
        X2_train = X2_train[indices]
        y_train = gap_train[indices]

        print X1_train.shape, X2_train.shape, X1_test.shape, X2_test.shape
        print y_train.shape, y_test.shape

        if compiler is None:
            compiler = Model_Compiler(X1_train.shape[2], hidden_dim=hidden_dim, n_class=n_class, fusion=fusion)
        compiler.compile(X1_train, X2_train, y_train, X1_test, X2_test, y_test)
        optimize(compiler.train_model, compiler.test_model, y_train.shape[0], y_test.shape[0])


def test1():
    cross_validation(n_class=2)


if __name__ == '__main__':
    test1()


