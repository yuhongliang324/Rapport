__author__ = 'yuhongliang324'

import sys
sys.path.append('..')
import theano
import theano.tensor as T
import numpy
from theano.tensor.shared_randomstreams import RandomStreams
from theano_utils import Adam, RMSprop, SGD, dropout


class RNN_Attention(object):
    # n_class = 1: regression problem
    # n_class > 1: classification problem
    # Change lamb to smaller value for hog
    # mlp_layers does not contain the input dim (depending on the model representation)
    # dec: whether or not use the decision GRU
    def __init__(self, input_dim, hidden_dim, mlp_layers, lamb=0., dec=True, update='adam',
                 drop=0.2, final_activation=None):
        self.input_dim, self.hidden_dim = input_dim, hidden_dim
        self.n_class = mlp_layers[-1]
        self.lamb = lamb
        self.dec = dec
        if self.dec:
            self.mlp_layers = [2 * hidden_dim] + mlp_layers
        else:
            self.mlp_layers = [input_dim] + mlp_layers
        self.drop = drop
        self.update = update
        self.rng = numpy.random.RandomState(1234)
        self.final_activation = final_activation
        theano_seed = numpy.random.randint(2 ** 30)
        self.theano_rng = RandomStreams(theano_seed)

        self.W_att_left_z, self.b_att_left_z, self.U_att_left_z,\
        self.W_att_left_r, self.b_att_left_r, self.U_att_left_r,\
        self.W_att_left_h, self.b_att_left_h, self.U_att_left_h,\
        self.W_att_right_z, self.b_att_right_z, self.U_att_right_z,\
        self.W_att_right_r, self.b_att_right_r, self.U_att_right_r,\
        self.W_att_right_h, self.b_att_right_h, self.U_att_right_h = self.create_gru_para()

        self.theta = [self.W_att_left_z, self.U_att_left_z, self.b_att_left_z,
                      self.W_att_left_r, self.U_att_left_r, self.b_att_left_r,
                      self.W_att_left_h, self.U_att_left_h, self.b_att_left_h,
                      self.W_att_right_z, self.U_att_right_z, self.b_att_right_z,
                      self.W_att_right_r, self.U_att_right_r, self.b_att_right_r,
                      self.W_att_right_h, self.U_att_right_h, self.b_att_right_h]

        if self.dec:
            self.W_dec_left_z, self.b_dec_left_z, self.U_dec_left_z,\
            self.W_dec_left_r, self.b_dec_left_r, self.U_dec_left_r,\
            self.W_dec_left_h, self.b_dec_left_h, self.U_dec_left_h,\
            self.W_dec_right_z, self.b_dec_right_z, self.U_dec_right_z,\
            self.W_dec_right_r, self.b_dec_right_r, self.U_dec_right_r,\
            self.W_dec_right_h, self.b_dec_right_h, self.U_dec_right_h = self.create_gru_para()

            self.theta += [self.W_dec_left_z, self.U_dec_left_z, self.b_dec_left_z,
                          self.W_dec_left_r, self.U_dec_left_r, self.b_dec_left_r,
                          self.W_dec_left_h, self.U_dec_left_h, self.b_dec_left_h,
                          self.W_dec_right_z, self.U_dec_right_z, self.b_dec_right_z,
                          self.W_dec_right_r, self.U_dec_right_r, self.b_dec_right_r,
                          self.W_dec_right_h, self.U_dec_right_h, self.b_dec_right_h]

        self.w_att = numpy.asarray(self.rng.uniform(
            low=-numpy.sqrt(6. / float(self.hidden_dim * 2 + 1)), high=numpy.sqrt(6. / float(self.hidden_dim * 2 + 1)),
            size=(self.hidden_dim * 2,)), dtype=theano.config.floatX)
        self.w_att = theano.shared(value=self.w_att, borrow=True)
        self.theta.append(self.w_att)

        '''
        self.W_1, self.b_1 = self.init_para(self.input_dim, self.n_class)
        self.theta += [self.W_1, self.b_1, self.W_s, self.U_s, self.b_s]'''

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

    def create_gru_para(self):
        W_left_z, b_left_z = self.init_para(self.input_dim, self.hidden_dim)
        U_left_z, _ = self.init_para(self.hidden_dim, self.hidden_dim)
        W_left_r, b_left_r = self.init_para(self.input_dim, self.hidden_dim)
        U_left_r, _ = self.init_para(self.hidden_dim, self.hidden_dim)
        W_left_h, b_left_h = self.init_para(self.input_dim, self.hidden_dim)
        U_left_h, _ = self.init_para(self.hidden_dim, self.hidden_dim)

        W_right_z, b_right_z = self.init_para(self.input_dim, self.hidden_dim)
        U_right_z, _ = self.init_para(self.hidden_dim, self.hidden_dim)
        W_right_r, b_right_r = self.init_para(self.input_dim, self.hidden_dim)
        U_right_r, _ = self.init_para(self.hidden_dim, self.hidden_dim)
        W_right_h, b_right_h = self.init_para(self.input_dim, self.hidden_dim)
        U_right_h, _ = self.init_para(self.hidden_dim, self.hidden_dim)
        return [W_left_z, b_left_z, U_left_z, W_left_r, b_left_r, U_left_r, W_left_h, b_left_h, U_left_h,
                W_right_z, b_right_z, U_right_z, W_right_r, b_right_r, U_right_r, W_right_h, b_right_h, U_right_h]

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

    def forward_att_GRU(self, X_t, H_tm1):
        Z_t = T.nnet.sigmoid(T.dot(X_t, self.W_att_left_z) + T.dot(H_tm1, self.U_att_left_z) + self.b_att_left_z)
        R_t = T.nnet.sigmoid(T.dot(X_t, self.W_att_left_r) + T.dot(H_tm1, self.U_att_left_r) + self.b_att_left_r)
        H_t = Z_t * H_tm1 + (1. - Z_t) * T.tanh(T.dot(X_t, self.W_att_left_h) + T.dot(R_t * H_tm1, self.U_att_left_h) +
                                                self.b_att_left_h)
        return H_t

    def backward_att_GRU(self, X_t, H_tm1):
        Z_t = T.nnet.sigmoid(T.dot(X_t, self.W_att_right_z) + T.dot(H_tm1, self.U_att_right_z) + self.b_att_right_z)
        R_t = T.nnet.sigmoid(T.dot(X_t, self.W_att_right_r) + T.dot(H_tm1, self.U_att_right_r) + self.b_att_right_r)
        H_t = Z_t * H_tm1 + (1. - Z_t) * T.tanh(T.dot(X_t, self.W_att_right_h) + T.dot(R_t * H_tm1, self.U_att_right_h) +
                                                self.b_att_right_h)
        return H_t

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

    def forward_attention(self, X_t, H_t_left, H_t_right, S_tm1):
        a_t = T.nnet.sigmoid(T.dot(T.concatenate((H_t_left, H_t_right), axis=1), self.w_att))
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
        H_att_foward, _ = theano.scan(self.forward_att_GRU, sequences=X_batch,
                                  outputs_info=T.zeros((batch_size, self.hidden_dim), dtype=theano.config.floatX))
        H_att_backward, _ = theano.scan(self.backward_att_GRU, sequences=X_batch[::-1],
                                    outputs_info=T.zeros((batch_size, self.hidden_dim), dtype=theano.config.floatX))
        H_att_backward = H_att_backward[::-1]

        H_att = T.concatenate([H_att_foward, H_att_backward], axis=2)  # (n_step, batch_size, 2 * hidden_dim)
        att = T.dot(H_att, self.w_att)  # (n_step, batch_size)
        att = T.nnet.softmax(att.T)  # (batch_size, n_step)
        if self.dec:
            H_dec_foward, _ = theano.scan(self.forward_dec_GRU, sequences=X_batch,
                                          outputs_info=T.zeros((batch_size, self.hidden_dim), dtype=theano.config.floatX))
            H_dec_backward, _ = theano.scan(self.backward_dec_GRU, sequences=X_batch[::-1],
                                            outputs_info=T.zeros((batch_size, self.hidden_dim), dtype=theano.config.floatX))
            H_dec_backward = H_dec_backward[::-1]
            H_tmp = T.concatenate([H_dec_foward, H_dec_backward], axis=2)  # (n_step, batch_size, 2 * hidden_dim)
            H_tmp = T.transpose(H_tmp, [1, 2, 0])  # (batch_size, 2 * hidden_dim, n_step)
        else:
            H_tmp = T.transpose(X_batch, [1, 2, 0])  # (batch_size, input_dim, n_step)
        rep = T.batched_dot(H_tmp, att)  # (batch_size, 2 * hidden_dim or input_dim)
        rep = H_att  # !!!
        '''
        [S, a], _ = theano.scan(self.forward_attention, sequences=[X_batch, H_foward, H_backward],
                                outputs_info=[T.zeros((batch_size, self.hidden_dim)), None])
        rep = S[-1]  # (batch_size, hidden_dim)'''

        if self.final_activation == 'tanh':
            rep = T.tanh(rep)
        elif self.final_activation == 'sigmoid':
            rep = T.nnet.sigmoid(rep)
        # rep = T.dot(rep, self.W_1) + self.b_1  # (batch_size, n_class)

        is_train = T.iscalar('is_train')
        numW = len(self.Ws)
        for i in xrange(numW - 1):
            rep = T.dot(rep, self.Ws[i]) + self.bs[i]
            '''
            if self.activation == 'relu':
                rep = T.maximum(rep, 0)
            elif self.activation == 'sigmoid':
                rep = T.nnet.sigmoid(rep)
            elif self.activation == 'softplus':
                rep = T.nnet.softplus(rep)
            else:
                rep = T.tanh(rep)'''
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
                'att': att, 'pred': pred, 'loss': loss, 'cost': cost, 'updates': updates}
        if self.n_class > 1:
            ret['acc'] = acc
        else:
            ret['loss_krip'] = loss_krip
        return ret
