__author__ = 'yuhongliang324'

from math import ceil, sqrt
import numpy
import theano
import theano.tensor as T
from sklearn.metrics import mean_squared_error

from rnn_attention import RNN_Attention
import sys
sys.path.append('../')
from model_utils import plot_loss
import argparse
import os
import shutil


def RMSE(y_actual, y_predicted):
    rmse = sqrt(mean_squared_error(y_actual, y_predicted))
    return rmse


def validate(test_model, y_test, costs_val, losses_krip_val, batch_size=32):
    n_test = y_test.shape[0]
    num_iter = int(ceil(n_test / float(batch_size)))
    all_pred = []
    cost_avg, loss_krip_avg = 0., 0.
    for iter_index in xrange(num_iter):
        start, end = iter_index * batch_size, min((iter_index + 1) * batch_size, n_test)
        cost, loss_krip, pred = test_model(start, end, 0)
        cost_avg += cost * (end - start)
        loss_krip_avg += loss_krip * (end - start)
        all_pred += pred.tolist()
    cost_avg /= n_test
    loss_krip_avg /= n_test
    costs_val.append(cost_avg)
    losses_krip_val.append(loss_krip_avg)
    rmse = RMSE(y_test, all_pred)
    print '\tTest cost = %f,\tKrip Loss = %f,\tRMSE = %f' % (cost_avg, loss_krip_avg, rmse)
    return cost_avg, all_pred


# model name can be added "-only" as suffix
def train(X_train, y_train, X_test, y_test, drop=0.25, final_activation=None,
          hidden_dim=None, batch_size=64, num_epoch=50): # !!!

    n_train = X_train.shape[0]
    input_dim = X_train.shape[2]
    X_train = X_train.transpose([1, 0, 2])
    X_test = X_test.transpose([1, 0, 2])

    X_train_shared = theano.shared(X_train, borrow=True)
    y_train_shared = theano.shared(y_train, borrow=True)
    X_test_shared = theano.shared(X_test, borrow=True)
    y_test_shared = theano.shared(y_test, borrow=True)

    ra = RNN_Attention(input_dim, hidden_dim, [hidden_dim, 1], dec=False,  # !!!
                       drop=drop, final_activation=final_activation)
    symbols = ra.build_model()

    X_batch, y_batch, is_train = symbols['X_batch'], symbols['y_batch'], symbols['is_train']
    att, pred, loss, loss_krip = symbols['att'], symbols['pred'], symbols['loss'], symbols['loss_krip']
    cost, updates = symbols['cost'], symbols['updates']

    num_iter = int(ceil(n_train / float(batch_size)))

    print 'Compiling function'

    start_symbol, end_symbol = T.lscalar(), T.lscalar()

    train_model = theano.function(inputs=[start_symbol, end_symbol, is_train],
                                  outputs=[cost, loss_krip, pred, att], updates=updates,
                                  givens={
                                      X_batch: X_train_shared[:, start_symbol: end_symbol, :],
                                      y_batch: y_train_shared[start_symbol: end_symbol]},
                                  on_unused_input='ignore', mode='FAST_RUN')
    print 'Compilation done 1'
    test_model = theano.function(inputs=[start_symbol, end_symbol, is_train],
                                  outputs=[cost, loss_krip, pred],
                                  givens={
                                      X_batch: X_test_shared[:, start_symbol: end_symbol, :],
                                      y_batch: y_test_shared[start_symbol: end_symbol]},
                                  on_unused_input='ignore', mode='FAST_RUN')
    print 'Compilation done 2'

    costs_train, costs_val = [], []
    losses_krip_train, losses_krip_val = [], []
    best_cost_val = 10000
    best_pred_val = None
    best_epoch = 0
    for epoch_index in xrange(num_epoch):
        cost_avg, loss_krip_avg, rmse = 0., 0., 0.
        all_pred = []
        print 'Epoch = %d' % (epoch_index + 1)
        for iter_index in xrange(num_iter):
            start, end = iter_index * batch_size, min((iter_index + 1) * batch_size, n_train)
            cost, loss_krip, pred, att = train_model(start, end, 1)
            cost_avg += cost * (end - start)
            loss_krip_avg += loss_krip * (end - start)
            all_pred += pred.tolist()
        cost_avg /= n_train
        loss_krip_avg /= n_train
        costs_train.append(cost_avg)
        losses_krip_train.append(loss_krip_avg)
        y_predicted = numpy.asarray(all_pred)
        rmse = RMSE(y_train, y_predicted)
        print '\tTrain cost = %f,\tKrip Loss = %f,\tRMSE = %f' % (cost_avg, loss_krip_avg, rmse)
        cost_avg_val, pred_val = validate(test_model, y_test, costs_val, losses_krip_val)
        if cost_avg_val < best_cost_val:
            best_cost_val = cost_avg_val
            best_pred_val = pred_val
            best_epoch = epoch_index
        if epoch_index - best_epoch >= 5 and epoch_index >= 30: # !!!
            return costs_train, costs_val, losses_krip_train, losses_krip_val, best_pred_val
    return costs_train, costs_val, losses_krip_train, losses_krip_val, best_pred_val


def cross_validation(feature_name='hog', side='b', drop=0., final_activation='sigmoid'):  # !!!

    feature_hidden = {'hog': 256, 'gemo': 128, 'au': 48, 'AU': 48, 'audio': 64}

    from data_preprocessing.load_data import load, load_audio
    from data_path import sample_10_root
    if feature_name == 'au' or feature_name == 'AU':
        tmp = 'gemo'
    else:
        tmp = feature_name
    # Use both speakers with adding features
    if feature_name == 'audio':
        dyad_features, dyad_ratings, dyad_slices = load_audio(side=side, normalization=False, best3=False)  # !!!
    else:
        dyad_features, dyad_ratings, dyad_slices = load(sample_10_root, feature_name=tmp, side=side)
    dyads = dyad_features.keys()
    hidden_dim = feature_hidden[feature_name]
    if feature_name == 'au' or feature_name == 'AU':
        for dyad, features in dyad_features.items():
            dyad_features[dyad] = features[:, :, -35:]
    num_dyad = len(dyads)
    message = 'attention_only_' + feature_name + '_' + side + '_drop_' + str(drop) + '_fact_' + str(final_activation)  # !!!
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
        costs_train, costs_val, losses_krip_train, losses_krip_val, best_pred_val\
            = train(X_train, y_train, X_test, y_test, hidden_dim=hidden_dim, drop=drop,
                    final_activation=final_activation)

        img_path = os.path.join(img_root, 'dyad_' + str(dyad) + '.png')
        plot_loss(img_path, costs_train, costs_val, dyad,
                  losses_krip_train=losses_krip_train, losses_krip_val=losses_krip_val)
        for i in xrange(y_test.shape[0]):
            writer.write(str(dyad) + ',' + str(slices_test[i][1]) + ',' + str(slices_test[i][2]) +
                         ',' + str(best_pred_val[i]) + ',' + str(y_test[i]) + '\n')
    writer.close()


def test1():
    parser = argparse.ArgumentParser()
    parser.add_argument('-feat', type=str, default='audio')
    parser.add_argument('-side', type=str, default=None)
    parser.add_argument('-drop', type=float, default=0.)
    parser.add_argument('-fact', type=str, default=None)
    args = parser.parse_args()
    if args.side is not None:
        side = args.side
    else:
        if args.feat == 'audio' or args.feat == 'au' or args.feat == 'AU':
            side = 'lr'
        else:
            side = 'ba'
    print args.feat, side
    cross_validation(feature_name=args.feat, side=side, drop=args.drop, final_activation=args.fact)


if __name__ == '__main__':
    test1()
