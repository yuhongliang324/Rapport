__author__ = 'yuhongliang324'

import sys
sys.path.append('..')
import theano
import theano.tensor as T
import numpy
from theano.tensor.shared_randomstreams import RandomStreams
from theano_utils import Adam, RMSprop, SGD, dropout


class dan(object):
    # n_class = 1: regression problem
    # n_class > 1: classification problem
    def __init__(self, layers, lamb=0.0002, update='adam', activation='relu', drop=0.5):
        self.layers = layers
        self.n_class = layers[-1]
        self.lamb = lamb
        self.drop = drop
        self.update = update
        self.activation = activation

        self.rng = numpy.random.RandomState(1234)
        theano_seed = numpy.random.randint(2 ** 30)
        self.theano_rng = RandomStreams(theano_seed)

        self.name = 'dan_' + '-'.join([str(x) for x in layers])

        self.theta = []
        self.Ws, self.bs = [], []

        num_layers = len(layers)
        for i in xrange(num_layers - 1):
            W, b = self.init_para(layers[i], layers[i + 1])
            self.Ws.append(W)
            self.bs.append(b)
            self.theta += [W, b]

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

    def build_model(self):
        X_batch = T.tensor3()  # (n_step, batch_size, input_dim)
        X_mean = T.mean(X_batch, axis=0)
        if self.n_class > 1:
            y_batch = T.ivector()  # (batch_size,)
        else:
            y_batch = T.vector()  # (batch_size,)

        is_train = T.iscalar('is_train')
        rep = X_mean
        numW = len(self.Ws)
        for i in xrange(numW - 1):
            rep = T.dot(rep, self.Ws[i]) + self.bs[i]
            rep = T.tanh(rep)
            if self.activation == 'relu':
                rep = T.maximum(rep, 0)
            elif self.activation == 'sigmoid':
                rep = T.nnet.sigmoid(rep)
            elif self.activation == 'softplus':
                rep = T.nnet.softplus(rep)
            else:
                rep = T.tanh(rep)
            rep = dropout(rep, is_train, drop_ratio=self.drop)

        batch_size = T.shape(y_batch)[0]

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
            loss = loss_sq / Z
        cost = loss + self.l2()
        updates = self.optimize(cost, self.theta)

        ret = {'X_batch': X_batch, 'y_batch': y_batch, 'is_train': is_train,
               'pred': pred, 'loss': loss, 'cost': cost, 'updates': updates}
        if self.n_class > 1:
            ret['acc'] = acc
        return ret
