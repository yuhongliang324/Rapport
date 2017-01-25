__author__ = 'yuhongliang324'
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import numpy


def Adam(cost, params, lr=0.0002, b1=0.1, b2=0.001, e=1e-8):
    updates = []
    grads = T.grad(cost, params)
    i = theano.shared(theano.config.floatX(0.))
    i_t = i + 1.
    fix1 = 1. - (1. - b1)**i_t
    fix2 = 1. - (1. - b2)**i_t
    lr_t = lr * (T.sqrt(fix2) / fix1)
    for p, g in zip(params, grads):
        m = theano.shared(p.get_value() * 0.)
        v = theano.shared(p.get_value() * 0.)
        m_t = (b1 * g) + ((1. - b1) * m)
        v_t = (b2 * T.sqr(g)) + ((1. - b2) * v)
        g_t = m_t / (T.sqrt(v_t) + e)
        p_t = p - (lr_t * g_t)
        updates.append((m, m_t))
        updates.append((v, v_t))
        updates.append((p, p_t))
    updates.append((i, i_t))
    return updates


def RMSprop(cost, params, lr=0.001, rho=0.9, epsilon=1e-6):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        acc = theano.shared(p.get_value() * 0.)
        acc_new = rho * acc + (1 - rho) * g ** 2
        gradient_scaling = T.sqrt(acc_new + epsilon)
        g = g / gradient_scaling
        updates.append((acc, acc_new))
        updates.append((p, p - lr * g))
    return updates


def SGD(cost, params, learning_rate=0.01):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for n, (param, grad) in enumerate(zip(params, grads)):
        updates.append((param, param - learning_rate * grad))
    return updates

theano_seed = numpy.random.randint(2 ** 30)
theano_rng = RandomStreams(theano_seed)


def dropout(layer, is_train, drop_ratio=0.5):
    remain = 1. - drop_ratio
    mask = theano_rng.binomial(p=remain, size=layer.shape, dtype=theano.config.floatX)
    return T.cast(T.switch(T.neq(is_train, 0), layer * mask, layer * remain), dtype=theano.config.floatX)
