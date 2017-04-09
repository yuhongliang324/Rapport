__author__ = 'yuhongliang324'

import sys
sys.path.append('..')
import theano
import theano.tensor as T
import numpy
from theano.tensor.shared_randomstreams import RandomStreams
from theano_utils import Adam, Adam2, RMSprop, SGD, dropout


class RNN(object):
    # n_class = 1: regression problem
    # n_class > 1: classification problem
    # Change lamb to smaller value for hog
    # mlp_layers does not contain the input dim (depending on the model representation)
    # dec: whether or not use the decision GRU
    def __init__(self, input_dim, hidden_dim, mlp_layers, lamb=0., model='gru', share=False, update='adam2',
                 drop=0.2, sq_loss=False):
        self.input_dim, self.hidden_dim = input_dim, hidden_dim
        self.n_class = mlp_layers[-1]
        self.lamb = lamb
        if model.endswith('l'):
            self.model = model[:-1]
            self.last = True
        else:
            self.model = model
            self.last = False
        self.share = share
        self.mlp_layers = [2 * hidden_dim] + mlp_layers
        self.drop = drop
        self.update = update
        self.sq_loss = sq_loss
        self.rng = numpy.random.RandomState(1234)
        theano_seed = numpy.random.randint(2 ** 30)
        self.theano_rng = RandomStreams(theano_seed)
        if self.model == 'lstm':
            self.W_dec_left_i, self.b_dec_left_i, self.U_dec_left_i, \
            self.W_dec_left_f, self.b_dec_left_f, self.U_dec_left_f, \
            self.W_dec_left_o, self.b_dec_left_o, self.U_dec_left_o, \
            self.W_dec_left_c, self.b_dec_left_c, self.U_dec_left_c = self.create_lstm_para()
            self.theta = [self.W_dec_left_i, self.b_dec_left_i, self.U_dec_left_i,
                          self.W_dec_left_f, self.b_dec_left_f, self.U_dec_left_f,
                          self.W_dec_left_o, self.b_dec_left_o, self.U_dec_left_o,
                          self.W_dec_left_c, self.b_dec_left_c, self.U_dec_left_c]
            if not share:
                self.W_dec_right_i, self.b_dec_right_i, self.U_dec_right_i, \
                self.W_dec_right_f, self.b_dec_right_f, self.U_dec_right_f, \
                self.W_dec_right_o, self.b_dec_right_o, self.U_dec_right_o, \
                self.W_dec_right_c, self.b_dec_right_c, self.U_dec_right_c = self.create_lstm_para()
                self.theta += [self.W_dec_right_i, self.b_dec_right_i, self.U_dec_right_i,
                               self.W_dec_right_f, self.b_dec_right_f, self.U_dec_right_f,
                               self.W_dec_right_o, self.b_dec_right_o, self.U_dec_right_o,
                               self.W_dec_right_c, self.b_dec_right_c, self.U_dec_right_c]
        else:
            self.W_dec_left_z, self.b_dec_left_z, self.U_dec_left_z, \
            self.W_dec_left_r, self.b_dec_left_r, self.U_dec_left_r, \
            self.W_dec_left_h, self.b_dec_left_h, self.U_dec_left_h = self.create_gru_para()
            self.theta = [self.W_dec_left_z, self.U_dec_left_z, self.b_dec_left_z,
                          self.W_dec_left_r, self.U_dec_left_r, self.b_dec_left_r,
                          self.W_dec_left_h, self.U_dec_left_h, self.b_dec_left_h]
            if not share:
                self.W_dec_right_z, self.b_dec_right_z, self.U_dec_right_z, \
                self.W_dec_right_r, self.b_dec_right_r, self.U_dec_right_r, \
                self.W_dec_right_h, self.b_dec_right_h, self.U_dec_right_h = self.create_gru_para()
                self.theta += [self.W_dec_right_z, self.U_dec_right_z, self.b_dec_right_z,
                               self.W_dec_right_r, self.U_dec_right_r, self.b_dec_right_r,
                               self.W_dec_right_h, self.U_dec_right_h, self.b_dec_right_h]

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

        print 'model =', self.model, 'lambda =', self.lamb, 'last =', self.last,\
            'share =', self.share, '#class =', self.n_class, 'drop =', self.drop, 'update =', self.update

        if self.update == 'adam':
            self.optimize = Adam
        elif self.update == 'adam2':
            self.optimize = Adam2
        elif self.update == 'rmsprop':
            self.optimize = RMSprop
        else:
            self.optimize = SGD

    def create_gru_para(self):
        W_z, b_z = self.init_para(self.input_dim, self.hidden_dim)
        U_z, _ = self.init_para(self.hidden_dim, self.hidden_dim)
        W_r, b_r = self.init_para(self.input_dim, self.hidden_dim)
        U_r, _ = self.init_para(self.hidden_dim, self.hidden_dim)
        W_h, b_h = self.init_para(self.input_dim, self.hidden_dim)
        U_h, _ = self.init_para(self.hidden_dim, self.hidden_dim)
        return [W_z, b_z, U_z, W_r, b_r, U_r, W_h, b_h, U_h]

    def create_lstm_para(self):
        W_i, b_i = self.init_para(self.input_dim, self.hidden_dim)
        U_i, _ = self.init_para(self.hidden_dim, self.hidden_dim)
        W_f, b_f = self.init_para(self.input_dim, self.hidden_dim)
        U_f, _ = self.init_para(self.hidden_dim, self.hidden_dim)
        W_o, b_o = self.init_para(self.input_dim, self.hidden_dim)
        U_o, _ = self.init_para(self.hidden_dim, self.hidden_dim)
        W_c, b_c = self.init_para(self.input_dim, self.hidden_dim)
        U_c, _ = self.init_para(self.hidden_dim, self.hidden_dim)
        return [W_i, b_i, U_i, W_f, b_f, U_f, W_o, b_o, U_o, W_c, b_c, U_c]

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

    def forward_dec_GRU(self, X_t, H_tm1):
        Z_t = T.nnet.sigmoid(T.dot(X_t, self.W_dec_left_z) + T.dot(H_tm1, self.U_dec_left_z) + self.b_dec_left_z)
        R_t = T.nnet.sigmoid(T.dot(X_t, self.W_dec_left_r) + T.dot(H_tm1, self.U_dec_left_r) + self.b_dec_left_r)
        H_t = Z_t * H_tm1 + (1. - Z_t) * T.tanh(T.dot(X_t, self.W_dec_left_h) + T.dot(R_t * H_tm1, self.U_dec_left_h) +
                                                self.b_dec_left_h)
        return H_t

    def backward_dec_GRU(self, X_t, H_tm1):
        Z_t = T.nnet.sigmoid(T.dot(X_t, self.W_dec_right_z) + T.dot(H_tm1, self.U_dec_right_z) + self.b_dec_right_z)
        R_t = T.nnet.sigmoid(T.dot(X_t, self.W_dec_right_r) + T.dot(H_tm1, self.U_dec_right_r) + self.b_dec_right_r)
        H_t = Z_t * H_tm1 + (1. - Z_t) * T.tanh(T.dot(X_t, self.W_dec_right_h) + T.dot(R_t * H_tm1, self.U_dec_right_h) +
                                                self.b_dec_right_h)
        return H_t

    def forward_dec_LSTM(self, X_t, C_tm1, H_tm1):
        i_t = T.nnet.sigmoid(T.dot(X_t, self.W_dec_left_i) + T.dot(H_tm1, self.U_dec_left_i) + self.b_dec_left_i)
        f_t = T.nnet.sigmoid(T.dot(X_t, self.W_dec_left_f) + T.dot(H_tm1, self.U_dec_left_f) + self.b_dec_left_f)
        o_t = T.nnet.sigmoid(T.dot(X_t, self.W_dec_left_o) + T.dot(H_tm1, self.U_dec_left_o) + self.b_dec_left_o)
        C_t = T.tanh(T.dot(X_t, self.W_dec_left_c) + T.dot(H_tm1, self.U_dec_left_c) + self.b_dec_left_c)
        C_t = i_t * C_t + f_t * C_tm1
        H_t = o_t * T.tanh(C_t)
        return C_t, H_t

    def backward_dec_LSTM(self, X_t, C_tm1, H_tm1):
        i_t = T.nnet.sigmoid(T.dot(X_t, self.W_dec_right_i) + T.dot(H_tm1, self.U_dec_right_i) + self.b_dec_right_i)
        f_t = T.nnet.sigmoid(T.dot(X_t, self.W_dec_right_f) + T.dot(H_tm1, self.U_dec_right_f) + self.b_dec_right_f)
        o_t = T.nnet.sigmoid(T.dot(X_t, self.W_dec_right_o) + T.dot(H_tm1, self.U_dec_right_o) + self.b_dec_right_o)
        C_t = T.tanh(T.dot(X_t, self.W_dec_right_c) + T.dot(H_tm1, self.U_dec_right_c) + self.b_dec_right_c)
        C_t = i_t * C_t + f_t * C_tm1
        H_t = o_t * T.tanh(C_t)
        return C_t, H_t

    def build_model(self):
        X_batch = T.tensor3()  # (n_step, batch_size, input_dim)
        s = T.shape(X_batch)
        if self.n_class > 1:
            y_batch = T.ivector()  # (batch_size,)
        else:
            y_batch = T.vector()  # (batch_size,)

        batch_size = T.shape(y_batch)[0]

        if self.model == 'lstm':
            [_, H_dec_forward], _ = theano.scan(self.forward_dec_LSTM, sequences=X_batch,
                                                outputs_info=[T.zeros((batch_size, self.hidden_dim),
                                                                      dtype=theano.config.floatX),
                                                              T.zeros((batch_size, self.hidden_dim),
                                                                      dtype=theano.config.floatX)])
            if self.share:
                [_, H_dec_backward], _ = theano.scan(self.forward_dec_LSTM, sequences=X_batch[::-1],
                                                     outputs_info=[T.zeros((batch_size, self.hidden_dim),
                                                                           dtype=theano.config.floatX),
                                                                   T.zeros((batch_size, self.hidden_dim),
                                                                           dtype=theano.config.floatX)])
            else:
                [_, H_dec_backward], _ = theano.scan(self.backward_dec_LSTM, sequences=X_batch[::-1],
                                                     outputs_info=[T.zeros((batch_size, self.hidden_dim),
                                                                           dtype=theano.config.floatX),
                                                                   T.zeros((batch_size, self.hidden_dim),
                                                                           dtype=theano.config.floatX)])
        else:
            H_dec_forward, _ = theano.scan(self.forward_dec_GRU, sequences=X_batch,
                                           outputs_info=[T.zeros((batch_size, self.hidden_dim),
                                                                 dtype=theano.config.floatX)])
            if self.share:
                H_dec_backward, _ = theano.scan(self.forward_dec_GRU, sequences=X_batch[::-1],
                                                outputs_info=[T.zeros((batch_size, self.hidden_dim),
                                                                      dtype=theano.config.floatX)])
            else:
                H_dec_backward, _ = theano.scan(self.backward_dec_GRU, sequences=X_batch[::-1],
                                                outputs_info=[T.zeros((batch_size, self.hidden_dim),
                                                                      dtype=theano.config.floatX)])
        H_dec_backward = H_dec_backward[::-1]
        H_tmp = T.concatenate([H_dec_forward, H_dec_backward], axis=2)  # (n_step, batch_size, 2 * hidden_dim)

        if self.last:
            rep = T.concatenate([H_dec_forward[-1], H_dec_backward[0]], axis=1)  # (batch_size, 2 * hidden_dim)
        else:
            rep = T.mean(H_tmp, axis=0)  # (batch_size, 2 * hidden_dim)

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
            if self.sq_loss:
                loss = loss_sq
                loss_krip = loss_sq
            else:
                # Z: 1/batch_size^2 * sum_{i,j} (pred_i - y_j)^2
                Z = batch_size * (T.sum(pred ** 2) + T.sum(y_batch ** 2)) - 2 * T.sum(T.outer(pred, y_batch))
                Z /= batch_size * batch_size
                loss_krip = loss_sq / Z
                loss = loss_krip
        cost = loss + self.l2()
        updates = self.optimize(cost, self.theta)

        ret = {'X_batch': X_batch, 'y_batch': y_batch, 'is_train': is_train,
               'att': None, 'pred': pred, 'loss': loss, 'cost': cost, 'updates': updates,
               'acc': None, 'loss_krip': None, 'rep': representation, 'shape': s}
        if self.n_class > 1:
            ret['acc'] = acc
            ret['prob'] = prob  # For computing AUC
        else:
            ret['loss_krip'] = loss_krip
        return ret
