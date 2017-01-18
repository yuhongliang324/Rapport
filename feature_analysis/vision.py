__author__ = 'yuhongliang324'
import sys
sys.path.append('..')
from data_preprocessing.load_data import load
from data_path import sample_10_root
import numpy
from sklearn.preprocessing import normalize


def get_PCC():
    dyad_features, dyad_ratings = load(sample_10_root, feature_name='gemo')
    dyads = dyad_features.keys()
    num_dyad = len(dyads)
    feature_list, rating_list = [], []
    for i in xrange(num_dyad):
        feature_list.append(dyad_features[dyads[i]])
        rating_list.append(dyad_ratings[dyads[i]])
    X = numpy.concatenate(feature_list)
    y = numpy.concatenate(rating_list)
    X_mean = numpy.mean(X, axis=1)
    X_max = numpy.max(X, axis=1)
    X_min = numpy.min(X, axis=1)

    def PCC(X, y, name):
        X_bar = normalize(X - numpy.mean(X, axis=0), axis=1)
        y_bar = normalize(y - numpy.mean(y))
        y_bar = numpy.squeeze(y_bar)
        PCC = numpy.dot(X_bar.T, y_bar)
        print name,
        print PCC.shape
        print PCC

    PCC(X_mean, y, 'mean')
    PCC(X_max, y, 'max')
    PCC(X_min, y, 'min')


if __name__ == '__main__':
    get_PCC()
