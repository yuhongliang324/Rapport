__author__ = 'yuhongliang324'

import numpy
from sklearn.feature_selection import SelectPercentile, f_classif


def select(X_list, ys, ratio, use_mean=False):
    def select1(X, ys, ratio=None):
        dim_origin = X.shape[-1]
        if ratio is None:
            dim = dim_origin // 2
        else:
            dim = int(dim_origin * ratio)
        selector = SelectPercentile(f_classif, percentile=10)
        selector.fit(X, ys)
        scores = -numpy.log10(selector.pvalues_)
        s = numpy.argsort(scores)
        s = s[::-1]
        ret = s[: dim]
        ret.sort()
        return ret

    xs = []
    for x in X_list:
        xs.append(numpy.mean(x, axis=0))
    xs = numpy.stack(xs)
    print select1(xs, ys, ratio)

    xs = numpy.concatenate(X_list, axis=0)
    yy = []
    for x, y in zip(X_list, ys):
        yy += [y] * x.shape[0]
    print select1(xs, ys, ratio)

