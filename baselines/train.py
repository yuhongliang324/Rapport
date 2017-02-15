__author__ = 'yuhongliang324'

from math import ceil, sqrt
import numpy
import theano
import theano.tensor as T
from sklearn.metrics import mean_squared_error

from dan import dan
import sys
sys.path.append('../')
from attention.model_utils import plot_loss
import argparse
import os
import shutil


def RMSE(y_actual, y_predicted):
    rmse = sqrt(mean_squared_error(y_actual, y_predicted))
    return rmse


def validate(test_model, y_test, costs_val, losses_val, batch_size=32):
    n_test = y_test.shape[0]
    num_iter = int(ceil(n_test / float(batch_size)))
    all_pred = []
    cost_avg, loss_avg = 0., 0.
    for iter_index in xrange(num_iter):
        start, end = iter_index * batch_size, min((iter_index + 1) * batch_size, n_test)
        cost, loss, pred = test_model(start, end, 0)
        cost_avg += cost * (end - start)
        loss_avg += loss * (end - start)
        all_pred += pred.tolist()
    cost_avg /= n_test
    loss_avg /= n_test
    costs_val.append(cost_avg)
    losses_val.append(loss_avg)
    rmse = RMSE(y_test, all_pred)
    print '\tTest cost = %f,\tLoss = %f,\tRMSE = %f' % (cost_avg, loss_avg, rmse)
    return cost_avg, all_pred


# model name can be added "-only" as suffix
def train(X_train, y_train, X_test, y_test, layers, activation='relu', drop=0.5, batch_size=64, num_epoch=30):

    n_train = X_train.shape[0]
    X_train = X_train.transpose([1, 0, 2])
    X_test = X_test.transpose([1, 0, 2])

    X_train_shared = theano.shared(X_train, borrow=True)
    y_train_shared = theano.shared(y_train, borrow=True)
    X_test_shared = theano.shared(X_test, borrow=True)
    y_test_shared = theano.shared(y_test, borrow=True)

    model = dan(layers, lamb=0., update='adam', activation=activation, drop=drop)
    symbols = model.build_model()

    X_batch, y_batch, is_train = symbols['X_batch'], symbols['y_batch'], symbols['is_train']
    att, pred, loss = symbols['a'], symbols['pred'], symbols['loss']
    cost, updates = symbols['cost'], symbols['updates']

    num_iter = int(ceil(n_train / float(batch_size)))

    print 'Compiling function'

    start_symbol, end_symbol = T.lscalar(), T.lscalar()

    train_model = theano.function(inputs=[start_symbol, end_symbol, is_train],
                                  outputs=[cost, loss, pred], updates=updates,
                                  givens={
                                      X_batch: X_train_shared[:, start_symbol: end_symbol, :],
                                      y_batch: y_train_shared[start_symbol: end_symbol]},
                                  on_unused_input='ignore', mode='FAST_RUN')
    print 'Compilation done 1'
    test_model = theano.function(inputs=[start_symbol, end_symbol, is_train],
                                  outputs=[cost, loss, pred],
                                  givens={
                                      X_batch: X_test_shared[:, start_symbol: end_symbol, :],
                                      y_batch: y_test_shared[start_symbol: end_symbol]},
                                  on_unused_input='ignore', mode='FAST_RUN')
    print 'Compilation done 2'

    costs_train, costs_val = [], []
    losses_train, losses_val = [], []
    best_cost_val = 10000
    best_pred_val = None
    best_epoch = 0
    for epoch_index in xrange(num_epoch):
        cost_avg, loss_avg, rmse = 0., 0., 0.
        all_pred = []
        print 'Epoch = %d' % (epoch_index + 1)
        for iter_index in xrange(num_iter):
            start, end = iter_index * batch_size, min((iter_index + 1) * batch_size, n_train)
            cost, loss, pred = train_model(start, end, 1)
            cost_avg += cost * (end - start)
            loss_avg += loss * (end - start)
            all_pred += pred.tolist()
        cost_avg /= n_train
        loss_avg /= n_train
        costs_train.append(cost_avg)
        losses_train.append(loss_avg)
        y_predicted = numpy.asarray(all_pred)
        rmse = RMSE(y_train, y_predicted)
        print '\tTrain cost = %f,\tLoss = %f,\tRMSE = %f' % (cost_avg, loss_avg, rmse)
        cost_avg_val, pred_val = validate(test_model, y_test, costs_val, losses_val)
        if cost_avg_val < best_cost_val:
            best_cost_val = cost_avg_val
            best_pred_val = pred_val
            best_epoch = epoch_index
        if epoch_index - best_epoch >= 5 and epoch_index >= 15:
            return costs_train, costs_val, losses_train, losses_val, best_pred_val
    return costs_train, costs_val, losses_train, losses_val, best_pred_val


def cross_validation(feature_name='hog', side='b', activation='relu', drop=0.5):

    fn_layers = {'hog': [256, 1], 'gemo': [128, 1], 'au': [48, 1], 'AU': [48, 1], 'audio': [64, 1]}

    from data_preprocessing.load_data import load, load_audio
    from data_path import sample_10_root
    if feature_name == 'au' or feature_name == 'AU':
        tmp = 'gemo'
    else:
        tmp = feature_name
    # Use both speakers with adding features
    if feature_name == 'audio':
        dyad_features, dyad_ratings, dyad_slices = load_audio(side=side)
    else:
        dyad_features, dyad_ratings, dyad_slices = load(sample_10_root, feature_name=tmp, side=side)
    dyads = dyad_features.keys()
    if feature_name == 'au' or feature_name == 'AU':
        for dyad, features in dyad_features.items():
            dyad_features[dyad] = features[:, :, -35:]
    input_dim = dyad_features[dyads[0]].shape[2]
    layers = [input_dim] + fn_layers[feature_name]
    num_dyad = len(dyads)
    message = 'dan_' + feature_name + '_' + side + '_drop_' + str(drop) + '_act_' + activation
    writer = open('../results/result_' + message + '.txt', 'w')
    img_root = '../figs/' + message
    if os.path.isdir(img_root):
        shutil.rmtree(img_root)
    os.mkdir(img_root)
    for i in xrange(num_dyad):
        dyad = dyads[i]
        X_test = dyad_features[dyad]
        y_test = dyad_ratings[dyad]
        slices_test = dyad_slices[dyad]
        feature_list, rating_list = [], []
        for j in xrange(num_dyad):
            if j == i:
                continue
            feature_list.append(dyad_features[dyads[j]])
            rating_list.append(dyad_ratings[dyads[j]])
        X_train = numpy.concatenate(feature_list)
        y_train = numpy.concatenate(rating_list)
        rating_mean = numpy.mean(y_train)
        rmse = y_test - rating_mean
        rmse = sqrt(numpy.mean(rmse * rmse))
        print 'Testing Dyad =', dyad
        print 'RMSE of Average Prediction = %f' % rmse
        print X_train.shape, X_test.shape
        costs_train, costs_val, losses_train, losses_val, best_pred_val\
            = train(X_train, y_train, X_test, y_test, layers, activation=activation, drop=drop)

        img_path = os.path.join(img_root, 'dyad_' + str(dyad) + '.png')
        plot_loss(img_path, costs_train, costs_val, dyad,
                  losses_krip_train=losses_train, losses_krip_val=losses_val)
        for i in xrange(y_test.shape[0]):
            writer.write(str(dyad) + ',' + str(slices_test[i][1]) + ',' + str(slices_test[i][2]) +
                         ',' + str(best_pred_val[i]) + ',' + str(y_test[i]) + '\n')
    writer.close()


def test1():
    parser = argparse.ArgumentParser()
    parser.add_argument('-feat', type=str, default='audio')
    parser.add_argument('-side', type=str, default=None)
    parser.add_argument('-drop', type=float, default=0.)
    parser.add_argument('-act', type=str, default=None)
    args = parser.parse_args()
    if args.side is not None:
        side = args.side
    else:
        if args.feat == 'audio' or args.feat == 'au' or args.feat == 'AU':
            side = 'b'
        else:
            side = 'ba'
    print args.feat, side
    cross_validation(feature_name=args.feat, side=side, activation=args.act, drop=args.drop)


if __name__ == '__main__':
    test1()