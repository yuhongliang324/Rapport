__author__ = 'yuhongliang324'

import sys
sys.path.append('../')
import argparse
from optimize import train
from utils import load_data, load_dict, train_pkl, test_pkl


def cross_validation(drop=0., hidden_dim=256,
                     dec=True, update='adam', lamb=0., model='ours', share=False, activation=None,
                     num_epoch=60, need_attention=False, sq_loss=False):

    X_train, y_train, start_batches_train, end_batches_train, len_batches_train\
        = load_data(train_pkl)
    X_test, y_test, start_batches_test, end_batches_test, len_batches_test\
        = load_data(test_pkl)
    _, E = load_dict()
    message = model + '_drop_' + str(drop) + '_lamb_' + str(lamb)
    if sq_loss:
        message += '_sq'
    result_file = 'results/' + message + '.txt'
    writer = open(result_file, 'w')

    print X_train.shape, X_test.shape

    best_pred_test, best_att_test \
        = train(E,
                X_train, y_train, start_batches_train, end_batches_train, len_batches_train,
                X_test, y_test, start_batches_test, end_batches_test, len_batches_test,
                drop=drop, dec=dec, update=update,
                hidden_dim=hidden_dim, num_epoch=num_epoch, lamb=lamb, model=model, share=share, category=True,
                activation=activation, need_attention=need_attention, sq_loss=sq_loss)

    for i in xrange(y_test.shape[0]):
        writer.write('%.4f' % best_pred_test[i] + ',' + str(y_test[i]) + '\n')
    writer.close()
    print 'Written to ' + result_file

    if need_attention:
        pass


def test1():
    parser = argparse.ArgumentParser()
    parser.add_argument('-drop', type=float, default=0.)
    parser.add_argument('-hid', type=int, default=256)
    parser.add_argument('-fact', type=str, default=None)
    parser.add_argument('-dec', type=int, default=1)
    parser.add_argument('-update', type=str, default='adam2')
    parser.add_argument('-lamb', type=float, default=0.)
    parser.add_argument('-model', type=str, default='ours')
    parser.add_argument('-share', type=int, default=0)
    parser.add_argument('-cat', type=int, default=0)
    parser.add_argument('-att', type=int, default=0)
    parser.add_argument('-sq', type=int, default=0)
    args = parser.parse_args()

    args.dec = bool(args.dec)
    args.share = bool(args.share)
    args.cat = bool(args.cat)
    args.att = bool(args.att)
    args.sq = bool(args.sq)

    activation = None
    if args.model == 'dan':
        activation = 'tanh'
    elif args.model == 'tagm':
        activation = 'relu'

    num_epoch = 20
    if args.model == 'dan':
        num_epoch = 50

    cross_validation(drop=args.drop, hidden_dim=args.hid,
                     dec=args.dec, update=args.update, lamb=args.lamb, model=args.model, share=args.share,
                     activation=activation,
                     num_epoch=num_epoch, need_attention=args.att, sq_loss=args.sq)


if __name__ == '__main__':
    test1()