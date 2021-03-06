__author__ = 'yuhongliang324'

from math import sqrt
import numpy
import sys
sys.path.append('../')
from model_utils import plot_loss
import argparse
import os
import shutil
from utils import load_split
from optimize import train
import cPickle
import random


def cross_validation(feature_name='hog', side='b', drop=0., final_activation=None, dec=True, update='adam', lamb=0.,
                     model='ours', share=False, best3=False, category=False, normalization=False, activation=None,
                     num_epoch=60, need_attention=False, draw=False, sq_loss=False, shuffle=False):

    feature_hidden = {'hog': 256, 'gemo': 128, 'au': 48, 'AU': 48, 'audio': 64}

    from data_preprocessing.load_data import load_vision, load_audio
    from data_path import sample_10_root

    if feature_name == 'audio':
        dyad_features, dyad_ratings, dyad_slices = load_audio(side=side, normalization=normalization,
                                                              best3=best3, category=category)
    else:
        dyad_features, dyad_ratings, dyad_slices = load_vision(sample_10_root, feature_name=feature_name,
                                                               side=side, normalization=normalization,
                                                               best3=best3, category=category)
    dyads = dyad_features.keys()
    hidden_dim = feature_hidden[feature_name]
    num_dyad = len(dyads)
    message = model + '_' + feature_name + '_' + side + '_share_' + str(share) + '_drop_' + str(drop)\
              + '_lamb_' + str(lamb) + '_fact_' + str(final_activation)
    if sq_loss:
        message += '_sq'
    if best3:
        message += '_best3'
    if category:
        message += '_cat'
    if shuffle:
        message += '_shuffle'
    result_file = '../results/result_' + message + '.txt'
    writer = open(result_file, 'w')

    all_slices = []
    all_attention = []

    dn = os.path.dirname(os.path.abspath(__file__))
    img_root = os.path.join(dn, '../figs/' + message)
    if draw:
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
            if dyads[j] == vdyad or dyads[j] == tdyad:
                continue
            feature_list.append(dyad_features[dyads[j]])
            rating_list.append(dyad_ratings[dyads[j]])
        X_train = numpy.concatenate(feature_list)
        y_train = numpy.concatenate(rating_list)
        if category:
            y_train = y_train.astype('int32')

        if shuffle:
            print 'Shuffling ...'
            n_train = X_train.shape[0]
            ind = numpy.arange(n_train)
            random.shuffle(ind)
            X_train = X_train[ind]
            y_train = y_train[ind]

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

        costs_train, costs_val, costs_test,\
        losses_krip_train, losses_krip_val, losses_krip_test, best_pred_test, best_att_test\
            = train(X_train, y_train, X_val, y_val, X_test, y_test, hidden_dim=hidden_dim, drop=drop,
                    final_activation=final_activation, dec=dec, update=update, lamb=lamb, model=model, share=share,
                    category=category, num_epoch=num_epoch, activation=activation, need_attention=need_attention,
                    sq_loss=sq_loss)

        if need_attention:
            all_slices += dyad_slices[tdyad]
            all_attention.append(best_att_test)

        if draw:
            img_path = os.path.join(img_root, 'dyad_' + str(vdyad) + '.png')
            if category:
                plot_loss(img_path, vdyad, costs_train, costs_val, costs_test=costs_test, tdyad=tdyad)
            else:
                plot_loss(img_path, vdyad, costs_train, costs_val,
                          losses_krip_train=losses_krip_train, losses_krip_val=losses_krip_val,
                          costs_test=costs_test, losses_krip_test=losses_krip_test, tdyad=tdyad)

        for i in xrange(y_test.shape[0]):
            writer.write(str(tdyad) + ',' + str(slices_test[i][1]) + ',' + str(slices_test[i][2]) +
                         ',' + '%.4f' % best_pred_test[i] + ',' + str(y_test[i]) + '\n')
    writer.close()
    print 'Written to ' + result_file

    if need_attention:
        all_attention = numpy.concatenate(all_attention, axis=0)
        pkl_path = feature_name + '_att.pkl'
        f = open(pkl_path, 'wb')
        cPickle.dump([all_slices, all_attention], f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()
        print 'Dump attention to ' + pkl_path


def test1():
    parser = argparse.ArgumentParser()
    parser.add_argument('-feat', type=str, default='audio')
    parser.add_argument('-side', type=str, default=None)
    parser.add_argument('-drop', type=float, default=0.)
    parser.add_argument('-fact', type=str, default=None)
    parser.add_argument('-dec', type=int, default=1)
    parser.add_argument('-update', type=str, default='adam2')
    parser.add_argument('-lamb', type=float, default=0.)
    parser.add_argument('-model', type=str, default='ours')
    parser.add_argument('-share', type=int, default=0)
    parser.add_argument('-cat', type=int, default=0)
    parser.add_argument('-best3', type=int, default=0)
    parser.add_argument('-att', type=int, default=0)
    parser.add_argument('-draw', type=int, default=0)
    parser.add_argument('-sq', type=int, default=0)
    parser.add_argument('-shuffle', type=int, default=0)
    args = parser.parse_args()
    if args.side is not None:
        side = args.side
    else:
        if args.feat == 'audio' or args.feat == 'gemo' or args.feat == 'au' or args.feat == 'AU':
            side = 'b'
        else:
            side = 'lr'
    if args.lamb >= 0:
        lamb = args.lamb
    else:
        if args.feat == 'audio' or args.feat == 'au' or args.feat == 'AU':
            lamb = 1e-4
        else:
            lamb = 5e-5
    print args.feat, side
    args.dec = bool(args.dec)
    args.share = bool(args.share)
    args.cat = bool(args.cat)
    args.best3 = bool(args.best3)
    args.att = bool(args.att)
    args.draw = bool(args.draw)
    args.sq = bool(args.sq)
    args.shuffle = bool(args.shuffle)

    normalization = False
    num_epoch = 40
    if args.model == 'tagm' and args.feat == 'audio':
        normalization = True
        args.update = 'rmsprop'
    if args.model == 'dan':
        num_epoch = 200
        args.update = 'adam'

    activation = None
    if args.model == 'tagm':
        if args.feat == 'hog':
            activation = 'relu'
        else:
            activation = 'softplus'
    elif args.model == 'dan':
        if args.feat == 'hog':
            activation = 'relu'
        else:
            activation = 'tanh'
    cross_validation(feature_name=args.feat, side=side, drop=args.drop, final_activation=args.fact,
                     dec=args.dec, update=args.update, lamb=lamb, model=args.model, share=args.share,
                     category=args.cat, best3=args.best3, normalization=normalization, activation=activation,
                     num_epoch=num_epoch, need_attention=args.att, draw=args.draw, sq_loss=args.sq, shuffle=args.shuffle)


if __name__ == '__main__':
    test1()
