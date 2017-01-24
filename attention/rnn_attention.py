__author__ = 'yuhongliang324'

import sys
sys.path.append('..')
import theano
import theano.tensor as T
import numpy
from theano.tensor.shared_randomstreams import RandomStreams
from optimizers import Adam, RMSprop, SGD


class RNN_Attention(object):
    # n_class = 1: regression problem
    # n_class > 1: classification problem
    def __init__(self, input_dim, hidden_dim, n_class, rnn='naive', lamb=0.00001, update='rmsprop'):
        self.rnn = rnn
        self.input_dim, self.hidden_dim = input_dim, hidden_dim
        self.n_class = n_class
        self.lamb = lamb
        self.update = update
        self.rng = numpy.random.RandomState(1234)
        theano_seed = numpy.random.randint(2 ** 30)
        self.theano_rng = RandomStreams(theano_seed)

        self.name = self.rnn + '_attention'

        if 'lstm' in self.rnn:
            self.W_i, self.b_i = self.init_para(self.input_dim, self.hidden_dim)
            self.U_i, _ = self.init_para(self.hidden_dim, self.hidden_dim)
            self.W_f, self.b_f = self.init_para(self.input_dim, self.hidden_dim)
            self.U_f, _ = self.init_para(self.hidden_dim, self.hidden_dim)
            self.W_o, self.b_o = self.init_para(self.input_dim, self.hidden_dim)
            self.U_o, _ = self.init_para(self.hidden_dim, self.hidden_dim)
            self.W_c, self.b_c = self.init_para(self.input_dim, self.hidden_dim)
            self.U_c, _ = self.init_para(self.hidden_dim, self.hidden_dim)

            self.theta = [self.W_i, self.U_i, self.b_i, self.W_f, self.U_f, self.b_f,
                          self.W_o, self.U_o, self.b_o, self.W_c, self.U_c, self.b_c]
        elif 'gru' in self.rnn:
            self.W_z, self.b_z = self.init_para(self.input_dim, self.hidden_dim)
            self.U_z, _ = self.init_para(self.hidden_dim, self.hidden_dim)
            self.W_r, self.b_r = self.init_para(self.input_dim, self.hidden_dim)
            self.U_r, _ = self.init_para(self.hidden_dim, self.hidden_dim)
            self.W_h, self.b_h = self.init_para(self.input_dim, self.hidden_dim)
            self.U_h, _ = self.init_para(self.hidden_dim, self.hidden_dim)
            self.theta = [self.W_z, self.U_z, self.b_z, self.W_r, self.U_r, self.b_r,
                          self.W_h, self.U_h, self.b_h]
        else:
            self.W_left, self.b_left = self.init_para(self.input_dim, self.hidden_dim)
            self.U_left, _ = self.init_para(self.hidden_dim, self.hidden_dim)
            self.theta = [self.W_left, self.U_left, self.b_left]

        self.W_s, self.b_s = self.init_para(self.input_dim, self.hidden_dim)
        self.U_s, _ = self.init_para(self.hidden_dim, self.hidden_dim)

        w_a = numpy.asarray(self.rng.uniform(
            low=-numpy.sqrt(6. / float(self.hidden_dim + 1)), high=numpy.sqrt(6. / float(self.hidden_dim + 1)),
            size=(self.hidden_dim,)), dtype=theano.config.floatX)
        self.w_a = theano.shared(value=w_a, borrow=True)

        self.W_1, self.b_1 = self.init_para(self.hidden_dim, self.n_class)

        self.theta += [self.w_a, self.W_1, self.b_1, self.W_s, self.U_s, self.b_s]

        if self.update == 'adam':
            self.optimize = Adam
        elif self.update == 'rmsprop':
            self.optimize = RMSprop
        else:
            self.optimize = SGD

    def init_para(self, d1, d2):
        W_values = numpy.asarray(self.rng.uniform(
            low=-numpy.sqrt(6. / float(d1 + d2)), high=numpy.sqrt(6. / float(d1 + d2)), size=(d1, d2)),
            dtype=theano.config.floatX)
        W = theano.shared(value=W_values, borrow=True)
        b_values = numpy.zeros((d2,), dtype=theano.config.floatX)
        b = theano.shared(value=b_values, borrow=True)
        return W, b

    def l2(self):
        l2 = self.lamb * T.sum([T.sum(p ** 2) for p in self.theta])
        return l2

    # scan function parameter order: sequences, prior results, non_sequences
    # sequences: X_t (batch_size, input_dim)
    # prior results: C_tm1, H_tm1, S_tm1 (batch_size, hidden_dim)
    def forward_LSTM(self, X_t, C_tm1, H_tm1, S_tm1):
        i_t = T.nnet.sigmoid(T.dot(X_t, self.W_i) + T.dot(H_tm1, self.U_i) + self.b_i)  # (batch_size, hidden_dim)
        f_t = T.nnet.sigmoid(T.dot(X_t, self.W_f) + T.dot(H_tm1, self.U_f) + self.b_f)  # (batch_size, hidden_dim)
        o_t = T.nnet.sigmoid(T.dot(X_t, self.W_o) + T.dot(H_tm1, self.U_o) + self.b_o)  # (batch_size, hidden_dim)
        C_t = T.tanh(T.dot(X_t, self.W_c) + T.dot(H_tm1, self.U_c) + self.b_c)  # (batch_size, hidden_dim)
        C_t = i_t * C_t + f_t * C_tm1  # (batch_size, hidden_dim)
        H_t = o_t * T.tanh(C_t)  # (batch_size, hidden_dim)
        a_t = T.nnet.sigmoid(T.dot(H_t, self.w_a))  # (batch_size,)
        S_t = T.tanh(T.dot(X_t, self.W_s) + T.dot(S_tm1, self.U_s) + self.b_s)  # (batch_size, hidden_dim)
        S_t = ((1. - a_t) * S_tm1.T + a_t * S_t.T).T
        return a_t, C_t, H_t, S_t

    def forward_GRU(self, X_t, H_tm1, S_tm1):
        z_t = T.nnet.sigmoid(T.dot(X_t, self.W_z) + T.dot(H_tm1, self.U_z) + self.b_z)
        r_t = T.nnet.sigmoid(T.dot(X_t, self.W_r) + T.dot(H_tm1, self.U_r) + self.b_r)
        H_t = T.tanh(T.dot(X_t, self.W_h) + T.dot(r_t * H_tm1, self.U_h) + self.b_h)
        H_t = (1 - z_t) * H_tm1 + z_t * H_t
        a_t = T.nnet.sigmoid(T.dot(H_t, self.w_a))
        S_t = T.tanh(T.dot(X_t, self.W_s) + T.dot(S_tm1, self.U_s) + self.b_s)
        S_t = ((1. - a_t) * S_tm1.T + a_t * S_t.T).T
        return a_t, H_t, S_t

    def forward_naive(self, X_t, H_tm1, S_tm1):
        H_t = T.nnet.softplus(T.dot(X_t, self.W_left) + T.dot(H_tm1, self.U_left) + self.b_left)
        a_t = T.nnet.sigmoid(T.dot(H_t, self.w_a))
        S_t = T.tanh(T.dot(X_t, self.W_s) + T.dot(S_tm1, self.U_s) + self.b_s)
        S_t = ((1. - a_t) * S_tm1.T + a_t * S_t.T).T
        return a_t, H_t, S_t

    def build_model(self):
        X_batch = T.tensor3()  # (n_step, batch_size, input_dim)
        if self.n_class > 1:
            y_batch = T.ivector()  # (batch_size,)
        else:
            y_batch = T.vector()  # (batch_size,)

        batch_size = T.shape(y_batch)[0]

        # a: (n_step, batch_size)
        # C: (n_step, batch_size, hidden_dim)
        # H: (n_step, batch_size, hidden_dim)

        if 'lstm' in self.rnn:
            [a, _, H, S], _ = theano.scan(self.forward_LSTM, sequences=X_batch,
                                          outputs_info=[None,
                                                        T.zeros((batch_size, self.hidden_dim), dtype=theano.config.floatX),
                                                        T.zeros((batch_size, self.hidden_dim), dtype=theano.config.floatX),
                                                        T.zeros((batch_size, self.hidden_dim), dtype=theano.config.floatX)])
        elif 'gru' in self.rnn:
            [a, H, S], _ = theano.scan(self.forward_GRU, sequences=X_batch,
                                       outputs_info=[None,
                                                     T.zeros((batch_size, self.hidden_dim), dtype=theano.config.floatX),
                                                     T.zeros((batch_size, self.hidden_dim), dtype=theano.config.floatX)])
        else:
            [a, H, S], _ = theano.scan(self.forward_naive, sequences=X_batch,
                                       outputs_info=[None,
                                                     T.zeros((batch_size, self.hidden_dim), dtype=theano.config.floatX),
                                                     T.zeros((batch_size, self.hidden_dim), dtype=theano.config.floatX)])
        if 'only' in self.rnn:
            rep = H[-1]
        else:
            rep = S[-1]  # (batch_size, hidden_dim)
        rep = T.dot(rep, self.W_1) + self.b_1  # (batch_size, n_class)
        if self.n_class > 1:
            prob = T.nnet.softmax(rep)[0]
            pred = T.argmax(prob)

            acc = T.mean(T.eq(pred, y_batch))

            loss = T.sum(-T.log(prob[y_batch]))
        else:
            pred = rep[:, 0]
            loss = pred - y_batch
            loss = T.mean(loss ** 2)
        cost = loss + self.l2()
        updates = self.optimize(cost, self.theta)
        ret = {'X_batch': X_batch, 'y_batch': y_batch,
                'a': a, 'pred': pred, 'loss': loss, 'cost': cost, 'updates': updates}
        if self.n_class > 1:
            ret['acc'] = acc
        return ret
