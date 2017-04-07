__author__ = 'yuhongliang324'

import sys
sys.path.append('..')
import theano
import theano.tensor as T
import numpy
from theano.tensor.shared_randomstreams import RandomStreams
from theano_utils import Adam, Adam2, AdaGrad, RMSprop, SGD, dropout


class TAGM(object):
    # n_class = 1: regression problem
    # n_class > 1: classification problem
    # Change lamb to smaller value for hog
    # mlp_layers does not contain the input dim (depending on the model representation)
    # dec: whether or not use the decision GRU
    def __init__(self, input_dim, hidden_dim, mlp_layers, lamb=0., update='adam2', drop=0.2):
        print 'TAGM'
        self.input_dim, self.hidden_dim = input_dim, hidden_dim
        self.n_class = mlp_layers[-1]
        self.lamb = lamb
        self.mlp_layers = [hidden_dim] + mlp_layers
        self.drop = drop
        self.update = update
        self.rng = numpy.random.RandomState(1234)
        theano_seed = numpy.random.randint(2 ** 30)
        self.theano_rng = RandomStreams(theano_seed)

        self.W_left, self.b_left = self.init_para(self.input_dim, self.hidden_dim)
        self.U_left, _ = self.init_para(self.hidden_dim, self.hidden_dim)
        self.W_right, self.b_right = self.init_para(self.input_dim, self.hidden_dim)
        self.U_right, _ = self.init_para(self.hidden_dim, self.hidden_dim)
        self.m, _ = self.init_para(self.hidden_dim * 2, 1)

        self.theta = [self.W_left, self.b_left, self.U_left,
                      self.W_right, self.b_right, self.U_right, self.m]

        self.W, self.b = self.init_para(self.input_dim, self.hidden_dim)
        self.U, _ = self.init_para(self.hidden_dim, self.hidden_dim)

        self.theta += [self.W, self.b, self.U]

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
        elif self.update == 'adam2':
            self.optimize = Adam2
        elif self.update == 'adagrad':
            self.optimize = AdaGrad
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

    def forward_attention(self, X_t, H_tm1):
        H_t = T.tanh(T.dot(X_t, self.W_left) + T.dot(H_tm1, self.U_left) + self.b_left)
        return H_t

    def backward_attention(self, X_t, H_tm1):
        H_t = T.tanh(T.dot(X_t, self.W_right) + T.dot(H_tm1, self.U_right) + self.b_right)
        return H_t

    # A_t: (batch_size,)
    def forward(self, X_t, A_t, H_tm1):
        H_tp = T.tanh(T.dot(X_t, self.W) + T.dot(H_tm1, self.U) + self.b)  # (batch_size, hid_size)
        H_t = ((1. - A_t) * H_tm1.T + A_t * H_tp.T).T
        return H_t

    def build_model(self):
        X_batch = T.tensor3()  # (n_step, batch_size, input_dim)
        if self.n_class > 1:
            y_batch = T.ivector()  # (batch_size,)
        else:
            y_batch = T.vector()  # (batch_size,)

        batch_size = T.shape(y_batch)[0]

        # both: (n_step, batch_size, hidden_dim)
        H_att_for, _ = theano.scan(self.forward_attention, sequences=X_batch,
                               outputs_info=[T.zeros((batch_size, self.hidden_dim), dtype=theano.config.floatX)])
        H_att_back, _ = theano.scan(self.backward_attention, sequences=X_batch[::-1],
                                    outputs_info=[T.zeros((batch_size, self.hidden_dim), dtype=theano.config.floatX)])
        H_att_back = H_att_back[::-1]  # (num_step, batch_size, hidden_dim)
        H_att = T.concatenate([H_att_for, H_att_back], axis=-1)  # (num_step, batch_size, 2 * hidden_dim)
        att = T.nnet.sigmoid(T.dot(H_att, self.m))  # (num_step, batch_size, 1)
        att = att[:, :, 0]  # (num_step, batch_size)
        # (num_step, batch_size, hidden_dim)
        H, _ = theano.scan(self.forward, sequences=[X_batch, att],
                           outputs_info=[T.zeros((batch_size, self.hidden_dim), dtype=theano.config.floatX)])
        rep = H[-1]
        is_train = T.iscalar('is_train')
        numW = len(self.Ws)
        for i in xrange(numW - 1):
            rep = T.dot(rep, self.Ws[i]) + self.bs[i]
            rep = T.tanh(rep)
            rep = dropout(rep, is_train, drop_ratio=self.drop)
        representation = rep
        rep = T.dot(rep, self.Ws[-1]) + self.bs[-1]
        rep = dropout(rep, is_train, drop_ratio=self.drop)  # (batch_size, num_class)

        if self.n_class > 1:
            prob = T.nnet.softmax(rep)
            pred = T.argmax(prob, axis=-1)

            acc = T.mean(T.eq(pred, y_batch))
            loss = T.mean(T.nnet.categorical_crossentropy(prob, y_batch))
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
               'pred': pred, 'loss': loss, 'cost': cost, 'updates': updates,
               'att': att, 'acc': None, 'loss_krip': None, 'rep': representation}
        if self.n_class > 1:
            ret['acc'] = acc
        else:
            ret['loss_krip'] = loss_krip
        return ret
