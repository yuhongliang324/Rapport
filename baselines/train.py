__author__ = 'yuhongliang324'

from math import sqrt
import numpy
import sys
sys.path.append('../')
import argparse
import os
import shutil
from utils import load_split
from optimize import train


def cross_validation(feature_name='hog', side='b', num_hidlayer=1, drop=0.5,
                     activation='relu', best3=False, category=False):

    fn_hid = {'hog': 256, 'gemo': 128, 'au': 48, 'AU': 48, 'audio': 64}

    from data_preprocessing.load_data import load_vision, load_audio
    from data_path import sample_10_root

    if feature_name == 'audio':
        dyad_features, dyad_ratings, dyad_slices = load_audio(side=side, normalization=False,
                                                              best3=best3, category=category)
    else:
        dyad_features, dyad_ratings, dyad_slices = load_vision(sample_10_root, feature_name=feature_name,
                                                               side=side, normalization=False,
                                                               best3=best3, category=category)

    dyads = dyad_features.keys()
    input_dim = dyad_features[dyads[0]].shape[2]
    layers = [input_dim]
    for i in xrange(num_hidlayer):
        layers.append(fn_hid[feature_name])
    if category:
        layers.append(3)
    else:
        layers.append(1)
    num_dyad = len(dyads)
    message = 'dan_' + feature_name + '_' + side + '_' + '-'.join([str(x) for x in layers])\
              + '_drop_' + str(drop) + '_act_' + activation
    if best3:
        message += '_best3'
    if category:
        message += '_cat'
    writer = open('../results/result_' + message + '.txt', 'w')
    dn = os.path.dirname(os.path.abspath(__file__))
    img_root = os.path.join(dn, '../figs/' + message)
    if os.path.isdir(img_root):
        shutil.rmtree(img_root)
    os.mkdir(img_root)
    vals, tests = load_split()
    print dyad_features.keys()
    for vdyad, tdyad in zip(vals, tests):
        X_val = dyad_features[vdyad]
        y_val = dyad_ratings[vdyad]
        if category:
            y_val = y_val.astype('int32')
        X_test = dyad_features[tdyad]
        y_test = dyad_ratings[tdyad]
        if category:
            y_test = y_test.astype('int32')
        slices_test = dyad_slices[tdyad]
        feature_list, rating_list = [], []
        for j in xrange(num_dyad):
            if j == vdyad or j == tdyad:
                continue
            feature_list.append(dyad_features[dyads[j]])
            rating_list.append(dyad_ratings[dyads[j]])
        X_train = numpy.concatenate(feature_list)
        y_train = numpy.concatenate(rating_list)
        if category:
            y_train = y_train.astype('int32')

        print 'Validation Dyad =', vdyad, '\tTesting Dyad =', tdyad
        if category:
            cnt = [0, 0, 0]
            for i in xrange(y_train.shape[0]):
                cnt[y_train[i]] += 1
            cnt = numpy.asarray(cnt)
            acc = numpy.max(cnt) / float(y_train.shape[0])
            cl = numpy.argmax(cnt)
            print 'Majority Accuracy = %f, Majority Class = %d' % (acc, cl)
        else:
            rating_mean = numpy.mean(y_train)
            rmse = y_val - rating_mean
            rmse = sqrt(numpy.mean(rmse * rmse))
            print 'RMSE of Average Prediction = %f' % rmse

        print X_train.shape, X_val.shape, X_test.shape

        costs_train, costs_val, costs_test, best_pred_test\
            = train(X_train, y_train, X_val, y_val, X_test, y_test, layers, drop=drop, category=category)

        for i in xrange(y_test.shape[0]):
            writer.write(str(tdyad) + ',' + str(slices_test[i][1]) + ',' + str(slices_test[i][2]) +
                         ',' + str(best_pred_test[i]) + ',' + str(y_test[i]) + '\n')
    writer.close()


def test1():
    parser = argparse.ArgumentParser()
    parser.add_argument('-feat', type=str, default='audio')
    parser.add_argument('-side', type=str, default=None)
    parser.add_argument('-drop', type=float, default=0.)
    parser.add_argument('-act', type=str, default='tanh')
    parser.add_argument('-cat', type=int, default=0)
    parser.add_argument('-best3', type=int, default=0)
    parser.add_argument('-nh', type=int, default=1)
    args = parser.parse_args()
    if args.side is not None:
        side = args.side
    else:
        if args.feat == 'audio' or args.feat == 'gemo' or args.feat == 'au' or args.feat == 'AU':
            side = 'b'
        else:
            side = 'lr'
    print args.feat, side
    args.cat = bool(args.cat)
    args.best3 = bool(args.best3)
    cross_validation(feature_name=args.feat, side=side, num_hidlayer=args.nh, drop=args.drop, activation=args.act,
                     category=args.cat, best3=args.best3)


if __name__ == '__main__':
    test1()
