__author__ = 'yuhongliang324'
import sys
import numpy
sys.path.append('../')
from utils import load_split
from math import sqrt
from optimize_ensemble import train


def cross_validation(dyad_slices, dyad_reps, dyad_ratings, message,
                     update='adam', lamb=0., drop=0., category=False, num_epoch=60):

    dyads = dyad_reps.keys()
    num_dyad = len(dyads)
    vals, tests = load_split()
    print dyad_reps.keys()
    writer = open('../results/' + message + '.txt', 'w')
    for vdyad, tdyad in zip(vals, tests):
        X_val = dyad_reps[vdyad]
        y_val = dyad_ratings[vdyad]
        if category:
            y_val = y_val.astype('int32')
        X_test = dyad_reps[tdyad]
        y_test = dyad_ratings[tdyad]
        if category:
            y_test = y_test.astype('int32')
        slices_test = dyad_slices[tdyad]
        feature_list, rating_list = [], []
        for j in xrange(num_dyad):
            if dyads[j] == vdyad or dyads[j] == tdyad:
                continue
            feature_list.append(dyad_reps[dyads[j]])
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

        costs_train, costs_val, costs_test,\
        losses_krip_train, losses_krip_val, losses_krip_test, best_pred_test\
            = train(X_train, y_train, X_val, y_val, X_test, y_test, drop=drop, update=update,
                    num_epoch=num_epoch, lamb=lamb, category=category)

        for i in xrange(y_test.shape[0]):
            writer.write(str(tdyad) + ',' + str(slices_test[i][1]) + ',' + str(slices_test[i][2]) +
                         ',' + str(best_pred_test[i]) + ',' + str(y_test[i]) + '\n')
    writer.close()
