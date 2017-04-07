__author__ = 'yuhongliang324'
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import numpy
from collections import OrderedDict


def Adam(cost, params, lr=0.0001, b1=0.1, b2=0.1, e=1e-8):
    """
    no bias init correction
    """
    updates = []
    grads = T.grad(cost, params)
    for p, g in zip(params, grads):
        m = theano.shared(p.get_value() * 0.)
        v = theano.shared(p.get_value() * 0.)
        m_t = (b1 * g) + ((1. - b1) * m)
        v_t = (b2 * T.sqr(g)) + ((1. - b2) * v)
        g_t = m_t / (T.sqrt(v_t) + e)
        p_t = p - (lr * g_t)
        updates.append((m, m_t))
        updates.append((v, v_t))
        updates.append((p, p_t))
    return updates


def Adam2(cost, params, lr=0.0002, b1=0.9, b2=0.999, e=1e-8):
    """
    no bias init correction
    """
    updates = []
    grads = T.grad(cost, params)
    for p, g in zip(params, grads):
        m = theano.shared(p.get_value() * 0.)
        v = theano.shared(p.get_value() * 0.)
        m_t = b1 * m + (1. - b1) * g
        v_t = b2 * v + (1. - b2) * T.sqr(g)
        g_t = m_t / (T.sqrt(v_t) + e)
        p_t = p - lr * g_t
        updates.append((m, m_t))
        updates.append((v, v_t))
        updates.append((p, p_t))
    return updates


def AdaGrad(cost, params, learning_rate=0.05, epsilon=1e-6):
    grads = T.grad(cost=cost, wrt=params)
    updates = OrderedDict()

    for param, grad in zip(params, grads):
        value = param.get_value(borrow=True)
        accu = theano.shared(numpy.zeros(value.shape, dtype=value.dtype),
                             broadcastable=param.broadcastable)
        accu_new = accu + grad ** 2
        updates[accu] = accu_new
        updates[param] = param - (learning_rate * grad /
                                  T.sqrt(accu_new + epsilon))

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


def SimpleSGD(cost, params, learning_rate=0.01):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for n, (param, grad) in enumerate(zip(params, grads)):
        updates.append((param, param - learning_rate * grad))
    return updates

theano_seed = numpy.random.randint(2 ** 30)
theano_rng = RandomStreams(theano_seed)


def SGD(cost, params, learning_rate=0.01, momentum=0.9):
    '''
    Compute updates for gradient descent with momentum

    :parameters:
        - cost : theano.tensor.var.TensorVariable
            Theano cost function to minimize
        - params : list of theano.tensor.var.TensorVariable
            Parameters to compute gradient against
        - learning_rate : float
            Gradient descent learning rate
        - momentum : float
            Momentum parameter, should be at least 0 (standard gradient descent) and less than 1

    :returns:
        updates : list
            List of updates, one for each parameter
    '''
    # Make sure momentum is a sane value
    assert 0 <= momentum < 1
    # List of update steps for each parameter
    updates = []
    # Just gradient descent on cost
    for param in params:
        # For each parameter, we'll create a previous_step shared variable.
        # This variable will keep track of the parameter's update step across iterations.
        # We initialize it to 0
        previous_step = theano.shared(param.get_value()*0., broadcastable=param.broadcastable)
        # Each parameter is updated by taking a step in the direction of the gradient.
        # However, we also "mix in" the previous step according to the given momentum value.
        # Note that we don't need to derive backpropagation to compute updates - just use T.grad!
        step = momentum*previous_step - learning_rate*T.grad(cost, param)
        # Add an update to store the previous step value
        updates.append((previous_step, step))
        # Add an update to apply the gradient descent step to the parameter itself
        updates.append((param, param + step))
    return updates


def dropout(layer, is_train, drop_ratio=0.5):
    remain = 1. - drop_ratio
    mask = theano_rng.binomial(p=remain, size=layer.shape, dtype=theano.config.floatX)
    return T.cast(T.switch(T.neq(is_train, 0), layer * mask, layer * remain), dtype=theano.config.floatX)
