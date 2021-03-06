__author__ = 'yuhongliang324'

import argparse
from math import sqrt

import numpy

from load_data import load_feature, pad
from mosi.optimize import train


def cross_validation(feature_name='audio', dec=True, update='adam', lamb=0., drop=0.,
                     model='gru', share=False, category=True, maxlen=1000, sample_rate=5):

    feature_hidden = {'video': 128, 'audio': 64}
    session_Xs, session_y = load_feature(feature_name=feature_name)
    sessions = session_Xs.keys()
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

    for test_session in sessions:
        Xs_test_list, y_test_list = session_Xs[test_session], session_y[test_session]
        Xs_test, y_test, start_batches_test, end_batches_test, len_batches_test\
            = pad(Xs_test_list, y_test_list, average=average, maxlen=maxlen, sample_rate=sample_rate)
        if category:
            y_test = y_test.astype('int32')
        Xs_train_list, y_train_list = [], []
        for session in sessions:
            if session == test_session:
                continue
            Xs_train_list += session_Xs[session]
            y_train_list += session_y[session]
        Xs_train, y_train, start_batches_train, end_batches_train, len_batches_train\
            = pad(Xs_train_list, y_train_list, average=average, maxlen=maxlen, sample_rate=sample_rate)
        if category:
            y_train = y_train.astype('int32')

        if category:
            cnt = [0, 0, 0, 0]
            for i in xrange(y_test.shape[0]):
                cnt[y_test[i]] += 1
            cnt = numpy.asarray(cnt)
            acc = numpy.max(cnt) / float(y_test.shape[0])
            cl = numpy.argmax(cnt)
            print 'Majority Accuracy = %f, Majority Class = %d' % (acc, cl)
        else:
            rating_mean = numpy.mean(y_train)
            rmse = y_test - rating_mean
            rmse = sqrt(numpy.mean(rmse * rmse))
            print 'RMSE of Average Prediction = %f' % rmse

        print Xs_train.shape, Xs_test.shape

        inputs_train = (Xs_train, y_train, start_batches_train, end_batches_train, len_batches_train)
        inputs_test = (Xs_test, y_test, start_batches_test, end_batches_test, len_batches_test)

        best_actual_test, best_pred_test\
            = train(inputs_train, inputs_test, hidden_dim=hidden_dim, dec=dec, update=update,
                    lamb=lamb, model=model, share=share, category=category, drop=drop, num_class=4)

        for i in xrange(y_test.shape[0]):
            writer.write(str(best_pred_test[i]) + ',' + str(best_actual_test[i]) + '\n')
    writer.close()


def test1():
    parser = argparse.ArgumentParser()
    parser.add_argument('-feat', type=str, default='audio')
    parser.add_argument('-dec', type=int, default=1)
    parser.add_argument('-update', type=str, default='adam2')
    parser.add_argument('-lamb', type=float, default=0.)
    parser.add_argument('-model', type=str, default='gru')
    parser.add_argument('-share', type=int, default=0)
    parser.add_argument('-cat', type=int, default=1)
    parser.add_argument('-maxlen', type=int, default=1000)
    parser.add_argument('-rate', type=int, default=5)
    parser.add_argument('-drop', type=float, default=0.)
    args = parser.parse_args()

    print args.feat
    args.dec = bool(args.dec)
    args.share = bool(args.share)
    args.cat = bool(args.cat)
    cross_validation(feature_name=args.feat, dec=args.dec, update=args.update, lamb=args.lamb, drop=args.drop,
                     model=args.model, share=args.share, category=args.cat, maxlen=args.maxlen, sample_rate=args.rate)


if __name__ == '__main__':
    test1()
