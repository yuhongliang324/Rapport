__author__ = 'yuhongliang324'

import sys
sys.path.append('..')
import theano
import theano.tensor as T
import numpy
from theano.tensor.shared_randomstreams import RandomStreams
from theano_utils import Adam, RMSprop, SGD, dropout


class LSTM(object):
    # n_class = 1: regression problem
    # n_class > 1: classification problem
    # Change lamb to smaller value for hog
    # mlp_layers does not contain the input dim (depending on the model representation)
    # dec: whether or not use the decision GRU
    def __init__(self, input_dim, hidden_dim, mlp_layers, lamb=0., update='adam',
                 drop=0.2):
        self.input_dim, self.hidden_dim = input_dim, hidden_dim
        self.n_class = mlp_layers[-1]
        self.lamb = lamb
        self.mlp_layers = [hidden_dim] + mlp_layers
        self.drop = drop
        self.update = update
        self.rng = numpy.random.RandomState(1234)
        theano_seed = numpy.random.randint(2 ** 30)
        self.theano_rng = RandomStreams(theano_seed)

        self.W_i, self.b_i = self.init_para(self.input_dim, self.hidden_dim)
        self.U_i, _ = self.init_para(self.hidden_dim, self.hidden_dim)
        self.W_f, self.b_f = self.init_para(self.input_dim, self.hidden_dim)
        self.U_f, _ = self.init_para(self.hidden_dim, self.hidden_dim)
        self.W_o, self.b_o = self.init_para(self.input_dim, self.hidden_dim)
        self.U_o, _ = self.init_para(self.hidden_dim, self.hidden_dim)
        self.W_c, self.b_c = self.init_para(self.input_dim, self.hidden_dim)
        self.U_c, _ = self.init_para(self.hidden_dim, self.hidden_dim)

        self.theta = [self.W_i, self.b_i, self.U_i, self.W_f, self.b_f, self.U_f, self.W_o, self.b_o, self.U_o,
                      self.W_c, self.b_c, self.U_c]

        self.Ws, self.bs = [], []
        num_layers = len(self.mlp_layers)
        for i in xrange(num_layers - 1):
            W, b = self.init_para(self.mlp_layers[i], self.mlp_layers[i + 1])
            self.Ws.append(W)
            self.bs.append(b)
        for W in self.Ws:
            self.theta.append(W)
        for b in self.bs:
            self.theta.append(b)

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

    def forward(self, X_t, C_tm1, H_tm1):
        i_t = T.nnet.sigmoid(T.dot(X_t, self.W_i) + T.dot(H_tm1, self.U_i) + self.b_i)
        f_t = T.nnet.sigmoid(T.dot(X_t, self.W_f) + T.dot(H_tm1, self.U_f) + self.b_f)
        o_t = T.nnet.sigmoid(T.dot(X_t, self.W_o) + T.dot(H_tm1, self.U_o) + self.b_o)
        C_t = T.tanh(T.dot(X_t, self.W_c) + T.dot(H_tm1, self.U_c) + self.b_c)
        C_t = i_t * C_t + f_t * C_tm1
        H_t = o_t * T.tanh(C_t)
        return C_t, H_t

    def build_model(self):
        X_batch = T.tensor3()  # (n_step, batch_size, input_dim)
        if self.n_class > 1:
            y_batch = T.ivector()  # (batch_size,)
        else:
            y_batch = T.vector()  # (batch_size,)

        batch_size = T.shape(y_batch)[0]

        # both: (n_step, batch_size, hidden_dim)
        [C, H], _ = theano.scan(self.forward, sequences=X_batch,
                                outputs_info=[T.zeros((batch_size, self.hidden_dim), dtype=theano.config.floatX),
                                              T.zeros((batch_size, self.hidden_dim), dtype=theano.config.floatX)])

        rep = H[-1]  # (batch_size, hidden_dim)

        is_train = T.iscalar('is_train')
        numW = len(self.Ws)
        for i in xrange(numW - 1):
            rep = T.dot(rep, self.Ws[i]) + self.bs[i]
            rep = T.tanh(rep)
            rep = dropout(rep, is_train, drop_ratio=self.drop)
        rep = T.dot(rep, self.Ws[-1]) + self.bs[-1]
        rep = dropout(rep, is_train, drop_ratio=self.drop)

        # rep = dropout(rep, is_train, drop_ratio=self.drop)

        if self.n_class > 1:
            prob = T.nnet.softmax(rep)[0]
            pred = T.argmax(prob)

            acc = T.mean(T.eq(pred, y_batch))

            loss = T.mean(-T.log(prob[y_batch]))
        else:
            pred = rep[:, 0]
            loss_sq = pred - y_batch
            loss_sq = T.mean(loss_sq ** 2)  # 1/batch_size (pred_i - y_i)^2
            # Z: 1/batch_size^2 * sum_{i,j} (pred_i - y_j)^2
            Z = batch_size * (T.sum(pred ** 2) + T.sum(y_batch ** 2)) - 2 * T.sum(T.outer(pred, y_batch))
            Z /= batch_size * batch_size
            loss_krip = loss_sq / Z
            loss = loss_krip
        cost = loss + self.l2()
        updates = self.optimize(cost, self.theta)

        ret = {'X_batch': X_batch, 'y_batch': y_batch, 'is_train': is_train,
                'pred': pred, 'loss': loss, 'cost': cost, 'updates': updates}
        if self.n_class > 1:
            ret['acc'] = acc
        else:
            ret['loss_krip'] = loss_krip
        return ret
