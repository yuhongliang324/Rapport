__author__ = 'yuhongliang324'

import numpy
from load_data import load, load_split
from sklearn.feature_selection import SelectPercentile, f_classif


def select(X_list, ys, ratio=None, use_mean=False):
    def select1(X, ys, ratio=None):
        X[numpy.isnan(X)] = 0.
        X[numpy.isneginf(X)] = 0.
        X[numpy.isinf(X)] = 0.
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

    if use_mean:
        xs = []
        for x in X_list:
            xs.append(numpy.mean(x, axis=0))
        xs = numpy.stack(xs)
        return select1(xs, ys, ratio=ratio)
    else:
        xs = numpy.concatenate(X_list, axis=0)
        yy = []
        for x, y in zip(X_list, ys):
            yy += [y] * x.shape[0]
        return select1(xs, yy, ratio)


def test1():
    feature_name = 'audio'
    category = True
    session_Xs, session_y = load(feature_name=feature_name, category=category)
    tests, trains = load_split()

    Xs_train_list, y_train_list = [], []
    for session in trains:
        if session not in session_Xs:
            continue
        Xs_train_list += session_Xs[session]
        y_train_list += session_y[session]
    select(Xs_train_list, y_train_list)


if __name__ == '__main__':
    test1()
