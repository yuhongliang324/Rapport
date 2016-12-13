__author__ = 'yuhongliang324'

import sys
sys.path.append('..')
import theano
import theano.tensor as T
import numpy
from theano.tensor.shared_randomstreams import RandomStreams
from collections import OrderedDict


class Bi_RNN_Attention(object):
    # n_class = 1: regression problem
    # n_class > 1: classification problem
    def __init__(self, input_dim, hidden_dim, n_class, rnn='naive', lamb=0.00001, update='rmsprop',
                 lr=None, beta1=0.9, beta2=0.999, epsilon=1e-8, decay=0., momentum=0.9, rho=0.9):
        self.rnn = rnn
        self.input_dim, self.hidden_dim = input_dim, hidden_dim
        self.n_class = n_class
        self.lamb = lamb
        self.update = update
        self.lr, self.momentum, self.rho = lr, momentum, rho
        self.beta1, self.beta2, self.epsilon, self.decay = beta1, beta2, epsilon, decay
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

        self.add_param_shapes()

        if self.update == 'adagrad':
            if lr:
                self.lr = lr
            else:
                self.lr = 0.01
            self.grad_histories = [
                theano.shared(
                    value=numpy.zeros(param_shape, dtype=theano.config.floatX),
                    borrow=True,
                    name="grad_hist:" + param.name
                )
                for param_shape, param in zip(self.param_shapes, self.theta)
                ]
        elif self.update == 'sgdm':
            if lr:
                self.lr = lr
            else:
                self.lr = 0.01
            self.velocity = [
                theano.shared(
                    value=numpy.zeros(param_shape, dtype=theano.config.floatX),
                    borrow=True,
                    name="momentum:" + param.name
                )
                for param_shape, param in zip(self.param_shapes, self.theta)
                ]
            self.momentum = momentum
        elif self.update == 'rmsprop' or self.update == 'RMSprop':
            self.rho = rho
            if lr:
                self.lr = lr
            else:
                self.lr = 0.001
            self.weights = [
                theano.shared(
                    value=numpy.zeros(param_shape, dtype=theano.config.floatX),
                    borrow=True,
                )
                for param_shape, param in zip(self.param_shapes, self.theta)
                ]
        elif self.update == 'adam' or self.update == 'Adam':
            pass
        else:  # sgd
            if lr:
                self.lr = lr
            else:
                self.lr = 0.01

    def init_para(self, d1, d2):
        W_values = numpy.asarray(self.rng.uniform(
            low=-numpy.sqrt(6. / float(d1 + d2)), high=numpy.sqrt(6. / float(d1 + d2)), size=(d1, d2)),
            dtype=theano.config.floatX)
        W = theano.shared(value=W_values, borrow=True)
        b_values = numpy.zeros((d2,), dtype=theano.config.floatX)
        b = theano.shared(value=b_values, borrow=True)
        return W, b

    def add_param_shapes(self):
        self.param_shapes = []
        for param in self.theta:
            self.param_shapes.append(param.get_value().shape)

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
        return a_t, S_t

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

        # (n_step, batch_size, hidden_dim)
        [a, _, S], _ = theano.scan(self.forward_naive, sequences=X_batch,
                                   outputs_info=[None,
                                                 T.zeros((batch_size, self.hidden_dim), dtype=theano.config.floatX),
                                                 T.zeros((batch_size, self.hidden_dim), dtype=theano.config.floatX)])
        [H_foward], _ = theano.scan(self.forward, sequences=X_batch,
                                    outputs_info=T.zeros((batch_size, self.hidden_dim), dtype=theano.config.floatX))
        [H_backward], _ = theano.scan(self.backward, sequences=X_batch[::-1],
                                      outputs_info=T.zeros((batch_size, self.hidden_dim), dtype=theano.config.floatX))
        H_backward = H_backward[::-1]
        [a, S], _ = theano.scan(self.forward_attention, sequences=[X_batch, H_foward, H_backward])

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
        gradients = [T.grad(cost, param) for param in self.theta]

        # adagrad
        if self.update == 'adagrad':
            new_grad_histories = [
                T.cast(g_hist + g ** 2, dtype=theano.config.floatX)
                for g_hist, g in zip(self.grad_histories, gradients)
                ]
            grad_hist_update = zip(self.grad_histories, new_grad_histories)

            param_updates = [(param, T.cast(param - self.lr / (T.sqrt(g_hist) + self.epsilon) * param_grad, dtype=theano.config.floatX))
                             for param, param_grad, g_hist in zip(self.theta, gradients, new_grad_histories)]
            updates = grad_hist_update + param_updates
        # SGD with momentum
        elif self.update == 'sgdm':
            velocity_t = [self.momentum * v + self.lr * g for v, g in zip(self.velocity, gradients)]
            velocity_updates = [(v, T.cast(v_t, theano.config.floatX)) for v, v_t in zip(self.velocity, velocity_t)]
            param_updates = [(param, T.cast(param - v_t, theano.config.floatX)) for param, v_t in zip(self.theta, velocity_t)]
            updates = velocity_updates + param_updates
        elif self.update == 'rmsprop' or self.update == 'RMSprop':
            updates = []
            for p, g, a in zip(self.theta, gradients, self.weights):
                # update accumulator
                new_a = self.rho * a + (1. - self.rho) * T.square(g)
                updates.append((a, new_a))
                new_p = p - self.lr * g / T.sqrt(new_a + self.epsilon)
                updates.append((p, new_p))
        # basic SGD
        else:
            updates = OrderedDict((p, T.cast(p - self.lr * g, dtype=theano.config.floatX)) for p, g in zip(self.theta, gradients))

        ret = {'X_batch': X_batch, 'y_batch': y_batch,
                'a': a, 'pred': pred, 'loss': loss, 'cost': cost, 'updates': updates}
        if self.n_class > 1:
            ret['acc'] = acc
        return ret
