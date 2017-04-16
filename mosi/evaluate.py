__author__ = 'yuhongliang324'

import numpy
from scipy.stats import pearsonr
import sys


def evaluate(fn):
    P = numpy.loadtxt(fn, delimiter=',')
    pred = P[:, 0]
    gt = P[:, 1]
    r = pearsonr(pred, gt)[0]
    return r


r = evaluate(sys.argv[1])
print 'Pearson = %f' % r
