__author__ = 'yuhongliang324'
import sys
sys.path.append('..')
from data_preprocessing.load_data import load
from data_path import sample_10_root
import numpy


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
        print X.shape, y.shape
