__author__ = 'yuhongliang324'

import argparse
import sys

from load_data import load, load_split

sys.path.append('../')
import numpy
from mosi.optimize import train
from IEMOCAP.load_data import pad
from feature_select import select


def experiment(feature_name='audio', dec=True, update='adam', lamb=0., drop=0., activation=None, sq_loss=False,
               model='ours', share=False, category=True, maxlen=1000, sample_rate=5,
               feat_sel=False, ratio=None, use_mean=False):

    feature_hidden = {'facet': 48, 'audio': 64, 'openface': 256, 'text': 128}
    session_Xs, session_y = load(feature_name=feature_name, category=category)
    hidden_dim = feature_hidden[feature_name]
    if dec:
        pref = 'ad'
    else:
        pref = 'att_only'
    message = pref + '_' + feature_name + '_model_' + model + '_share_' + str(share) + '_lamb_' + str(lamb)\
              + '_drop_' + str(drop)
    if category:
        message += '_cat'
    writer = open('results/result_' + message + '.txt', 'w')

    if feature_name == 'audio':
        average = True
    else:
        average = False

    tests, trains = load_split()

    Xs_test_list, y_test_list = [], []
    for session in tests:
        if session not in session_Xs:
            continue
        Xs_test_list += session_Xs[session]
        y_test_list += session_y[session]

    Xs_train_list, y_train_list = [], []
    for session in trains:
        if session not in session_Xs:
            continue
        Xs_train_list += session_Xs[session]
        y_train_list += session_y[session]

    if feat_sel:
        feature_indices = select(Xs_train_list, y_train_list, ratio=ratio, use_mean=use_mean)
        Xs_train_list = map(lambda x: x[:, feature_indices], Xs_train_list)
        Xs_test_list = map(lambda x: x[:, feature_indices], Xs_test_list)
        hidden_dim = int(hidden_dim * ratio)

    Xs_test, y_test, start_batches_test, end_batches_test, len_batches_test \
        = pad(Xs_test_list, y_test_list, average=average, maxlen=maxlen, sample_rate=sample_rate)
    if category:
        y_test = y_test.astype('int32')

    Xs_train, y_train, start_batches_train, end_batches_train, len_batches_train \
        = pad(Xs_train_list, y_train_list, average=average, maxlen=maxlen, sample_rate=sample_rate)
    if category:
        y_train = y_train.astype('int32')

    if category:
        cnt = [0, 0]
        for i in xrange(y_test.shape[0]):
            cnt[y_test[i]] += 1
        cnt = numpy.asarray(cnt)
        acc = numpy.max(cnt) / float(y_test.shape[0])
        cl = numpy.argmax(cnt)
        print 'Majority Accuracy = %f, Majority Class = %d' % (acc, cl)
    else:
        rating_mean = numpy.mean(y_train)
        mae = numpy.abs(y_test - rating_mean)
        mae = numpy.mean(mae)
        print 'MAE of Average Prediction = %f' % mae

    print Xs_train.shape, Xs_test.shape

    inputs_train = (Xs_train, y_train, start_batches_train, end_batches_train, len_batches_train)
    inputs_test = (Xs_test, y_test, start_batches_test, end_batches_test, len_batches_test)

    best_actual_test, best_pred_test \
        = train(inputs_train, inputs_test, hidden_dim=hidden_dim, dec=dec, update=update,
                activation=activation, sq_loss=sq_loss, num_epoch=20,
                lamb=lamb, model=model, share=share, category=category, drop=drop, num_class=2)

    for i in xrange(y_test.shape[0]):
        writer.write(str(best_pred_test[i]) + ',' + str(best_actual_test[i]) + '\n')
    writer.close()


def test1():
    parser = argparse.ArgumentParser()
    parser.add_argument('-feat', type=str, default='audio')
    parser.add_argument('-dec', type=int, default=1)
    parser.add_argument('-update', type=str, default='adam2')
    parser.add_argument('-lamb', type=float, default=0.)
    parser.add_argument('-model', type=str, default='ours')
    parser.add_argument('-share', type=int, default=0)
    parser.add_argument('-cat', type=int, default=1)
    parser.add_argument('-maxlen', type=int, default=1000)
    parser.add_argument('-rate', type=int, default=2)
    parser.add_argument('-drop', type=float, default=0.)
    parser.add_argument('-sq', type=int, default=0)
    parser.add_argument('-fs', type=int, default=0)
    parser.add_argument('-fs_ratio', type=float, default=0.5)
    parser.add_argument('-um', type=int, default=0)
    args = parser.parse_args()

    print args.feat
    args.dec = bool(args.dec)
    args.share = bool(args.share)
    args.cat = bool(args.cat)
    args.sq = bool(args.sq)
    args.fs = bool(args.fs)
    args.um = bool(args.um)

    if args.feat == 'audio':
        args.rate = 3
    elif args.feat == 'text':
        args.rate = 1

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

    experiment(feature_name=args.feat, dec=args.dec, update=args.update, lamb=args.lamb, drop=args.drop,
               activation=activation, sq_loss=args.sq, model=args.model, share=args.share, category=args.cat,
               maxlen=args.maxlen, sample_rate=args.rate, feat_sel=args.fs, ratio=args.fs_ratio, use_mean=args.um)


if __name__ == '__main__':
    test1()
