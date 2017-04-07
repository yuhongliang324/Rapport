__author__ = 'yuhongliang324'

from math import sqrt
import numpy
import sys
sys.path.append('../')
import argparse
from utils import load_split
from optimize_single import train
from collections import defaultdict
from cross_validation_ensemble import cross_validation as cv2
import theano


def cross_validation(feature_name='hog', side='b', drop=0., final_activation=None, dec=True, update='adam', lamb=0.,
                     model='ours', share=False, best3=False, category=False, normalization=False, num_epoch=60):
    print feature_name, side
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
    message = model + '_' + feature_name.replace(' ', '+') + '_' + side + '_share_' + str(share) + '_drop_' + str(drop)\
              + '_lamb_' + str(lamb) + '_fact_' + str(final_activation)
    if best3:
        message += '_best3'
    if category:
        message += '_cat'
    vals, tests = load_split()
    print dyad_features.keys()
    dyad_rep = {}
    for vdyad, tdyad in zip(vals, tests):
        X_val = dyad_features[vdyad]
        y_val = dyad_ratings[vdyad]
        if category:
            y_val = y_val.astype('int32')
        X_test = dyad_features[tdyad]
        y_test = dyad_ratings[tdyad]
        if category:
            y_test = y_test.astype('int32')
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

        _, _, _, _, _, _, best_rep_test\
            = train(X_train, y_train, X_val, y_val, X_test, y_test, hidden_dim=hidden_dim, drop=drop,
                    final_activation=final_activation, dec=dec, update=update, lamb=lamb, model=model, share=share,
                    category=category, num_epoch=num_epoch)
        dyad_rep[tdyad] = best_rep_test
    return dyad_slices, dyad_rep, dyad_ratings


def slice_unite(slice_tup):
    ret = (int(slice_tup[0]), int(slice_tup[1]), int(slice_tup[2]))
    return ret


def test1():
    parser = argparse.ArgumentParser()
    parser.add_argument('-feat', type=str, default='audio hog')
    parser.add_argument('-drop', type=float, default=0.)
    parser.add_argument('-fact', type=str, default=None)
    parser.add_argument('-dec', type=int, default=1)
    parser.add_argument('-update', type=str, default='adam')
    parser.add_argument('-lamb', type=float, default=0.)
    parser.add_argument('-model', type=str, default='ours')
    parser.add_argument('-share', type=int, default=0)
    parser.add_argument('-cat', type=int, default=0)
    parser.add_argument('-best3', type=int, default=0)
    args = parser.parse_args()
    args.dec = bool(args.dec)
    args.share = bool(args.share)
    args.cat = bool(args.cat)
    args.best3 = bool(args.best3)
    normalization = False
    if args.model == 'tagm' or args.model == 'dan':
        normalization = True

    dyad_slices_all, dyad_rep_all, dyad_ratings_all = {}, {}, {}
    for feat in args.feat.split():
        num_epoch = 60
        if args.model == 'dan':
            num_epoch = 200
        dyad_slices, dyad_rep, dyad_ratings = \
            cross_validation(feature_name=feat, side='ba', drop=args.drop, final_activation=args.fact,
                             dec=args.dec, update=args.update, lamb=args.lamb, model=args.model, share=args.share,
                             category=args.cat, best3=args.best3, normalization=normalization, num_epoch=num_epoch)
        for dyad, slices in dyad_slices.items():
            if dyad not in dyad_slices_all:
                dyad_slices_all[dyad] = []
            dyad_slices_all[dyad].append(slices)
        for dyad, rep in dyad_rep.items():
            if dyad not in dyad_rep_all:
                dyad_rep_all[dyad] = []
            dyad_rep_all[dyad].append(rep)
        for dyad, ratings in dyad_ratings.items():
            if dyad not in dyad_ratings_all:
                dyad_ratings_all[dyad] = []
            dyad_ratings_all[dyad].append(ratings)

    dyad_slices_ensemble = {}
    dyad_reps_ensemble = {}
    dyad_ratings_ensemble = {}
    num_modal = len(args.feat.split())
    dyads = dyad_slices_all.keys()
    for dyad in dyads:
        slices_all = dyad_slices_all[dyad]
        rep_all = dyad_rep_all[dyad]
        ratings_all = dyad_ratings_all[dyad]
        slice_count = defaultdict(int)
        slice_rep = {}
        slice_rating = {}
        for slices, reps, ratings in zip(slices_all, rep_all, ratings_all):
            for slice, rep, rating in zip(slices, reps, ratings):
                slice = slice_unite(slice)
                if slice not in slice_rep:
                    slice_rep[slice] = []
                if slice not in slice_rating:
                    slice_rating[slice] = []
                slice_rep[slice].append(rep)
                slice_rating[slice].append(rating)
                slice_count[slice] += 1
        slices_ensemble = []
        reps_ensemble = []
        ratings_ensemble = []
        for slice, count in slice_count.items():
            if count < num_modal:
                continue
            slices_ensemble.append(slice)
            enRating = slice_rating[slice][0]
            enRep = numpy.concatenate(slice_rep[slice])
            reps_ensemble.append(enRep)
            ratings_ensemble.append(enRating)
        if len(slices_ensemble) < 10:
            continue
        dyad_slices_ensemble[dyad] = slices_ensemble
        dyad_reps_ensemble[dyad] = numpy.stack(reps_ensemble, axis=0)
        dyad_ratings_ensemble[dyad] = numpy.asarray(ratings_ensemble, dtype=theano.config.floatX)

    message = 'ensemble_' + args.model + '_' + args.feat + '_drop_' + str(args.drop) + '_lamb_' + str(args.lamb)

    cv2(dyad_slices_ensemble, dyad_reps_ensemble, dyad_ratings_ensemble,
        message, update=args.update, lamb=args.lamb, drop=args.drop, category=args.cat, num_epoch=20)

if __name__ == '__main__':
    test1()
