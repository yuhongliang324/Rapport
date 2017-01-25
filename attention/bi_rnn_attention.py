__author__ = 'yuhongliang324'

import sys
sys.path.append('..')
import theano
import theano.tensor as T
import numpy
from theano.tensor.shared_randomstreams import RandomStreams
from optimizers import Adam, RMSprop, SGD


class Bi_RNN_Attention(object):
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

        self.name = 'bi-' + self.rnn + '_attention'

        self.W_left, self.b_left = self.init_para(self.input_dim, self.hidden_dim)
        self.U_left, _ = self.init_para(self.hidden_dim, self.hidden_dim)
        self.W_right, self.b_right = self.init_para(self.input_dim, self.hidden_dim)
        self.U_right, _ = self.init_para(self.hidden_dim, self.hidden_dim)
        self.theta = [self.W_left, self.U_left, self.b_left, self.W_right, self.U_right, self.b_right]

        self.W_s, self.b_s = self.init_para(self.input_dim, self.hidden_dim)
        self.U_s, _ = self.init_para(self.hidden_dim, self.hidden_dim)

        w_a = numpy.asarray(self.rng.uniform(
            low=-numpy.sqrt(6. / float(self.hidden_dim * 2 + 1)), high=numpy.sqrt(6. / float(self.hidden_dim * 2 + 1)),
            size=(self.hidden_dim * 2,)), dtype=theano.config.floatX)
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

    def forward(self, X_t, H_tm1):
        H_t = T.nnet.softplus(T.dot(X_t, self.W_left) + T.dot(H_tm1, self.U_left) + self.b_left)
        return H_t

    def backward(self, X_t, H_tm1):
        H_t = T.nnet.softplus(T.dot(X_t, self.W_right) + T.dot(H_tm1, self.U_right) + self.b_right)
        return H_t

    def forward_attention(self, X_t, H_t_left, H_t_right, S_tm1):
        a_t = T.nnet.sigmoid(T.dot(T.concatenate((H_t_left, H_t_right), axis=1), self.w_a))
        S_t = T.tanh(T.dot(X_t, self.W_s) + T.dot(S_tm1, self.U_s) + self.b_s)
        S_t = ((1. - a_t) * S_tm1.T + a_t * S_t.T).T
        return S_t, a_t

    def build_model(self):
        X_batch = T.tensor3()  # (n_step, batch_size, input_dim)
        if self.n_class > 1:
            y_batch = T.ivector()  # (batch_size,)
        else:
            y_batch = T.vector()  # (batch_size,)

        batch_size = T.shape(y_batch)[0]

        # (n_step, batch_size, hidden_dim)
        [H_foward], _ = theano.scan(self.forward, sequences=X_batch,
                                       outputs_info=[T.alloc(theano.config.floatX, batch_size, self.hidden_dim)])
        [H_backward], _ = theano.scan(self.backward, sequences=X_batch[::-1],
                                         outputs_info=[T.alloc(theano.config.floatX, batch_size, self.hidden_dim)])
        H_backward = H_backward[::-1]
        [S, a], _ = theano.scan(self.forward_attention, sequences=[X_batch, H_foward, H_backward],
                                  outputs_info=[T.zeros((batch_size, self.hidden_dim)), None])

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
