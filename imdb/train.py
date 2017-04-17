__author__ = 'yuhongliang324'

import sys
sys.path.append('../')
import argparse
from optimize import train
from utils import load_data, load_dict, train_pkl, test_pkl
import cPickle


def cross_validation(drop=0., hidden_dim=256,
                     dec=True, update='adam', lamb=0., model='ours', share=False, activation=None,
                     num_epoch=60, need_attention=False, sq_loss=False, fine=False):

    X_train, y_train, start_batches_train, end_batches_train, len_batches_train, _\
        = load_data(train_pkl, fine=fine)
    X_test, y_test, start_batches_test, end_batches_test, len_batches_test, indices_test\
        = load_data(test_pkl, fine=fine)
    token_ID, E = load_dict()
    message = model + '_drop_' + str(drop) + '_lamb_' + str(lamb) + '_hid_' + str(hidden_dim)
    if sq_loss:
        message += '_sq'
    if fine:
        message += '_fine'
    if share:
        message += '_share'
    result_file = 'results/' + message + '.txt'
    writer = open(result_file, 'w')

    print E.shape, X_train.shape, X_test.shape

    num_class = 2
    if fine:
        num_class = 8
    best_pred_test, best_att_test \
        = train(E,
                X_train, y_train, start_batches_train, end_batches_train, len_batches_train,
                X_test, y_test, start_batches_test, end_batches_test, len_batches_test,
                drop=drop, dec=dec, update=update,
                hidden_dim=hidden_dim, num_epoch=num_epoch, lamb=lamb, model=model, share=share, category=True,
                activation=activation, need_attention=need_attention, sq_loss=sq_loss, num_class=num_class)

    for i in xrange(y_test.shape[0]):
        writer.write(str(indices_test[i]) + ',' + '%.4f' % best_pred_test[i] + ',' + str(y_test[i]) + '\n')
    writer.close()
    print 'Written to ' + result_file

    if need_attention:
        ID_token = {0: '*UNKNOWN*'}
        for token, ID in token_ID.items():
            ID_token[ID] = token

        num_iter = len(start_batches_test)
        word_indices = []
        for iter_index in xrange(num_iter):
            start, end = start_batches_train[iter_index], end_batches_train[iter_index]
            length = len_batches_train[iter_index]
            xb = X_train[start: end, :length]
            word_indices.append(xb)

        sentences, attentions = [], []
        for xb, atts in zip(word_indices, best_att_test):
            size = xb.shape[0]
            for i in xrange(size):
                x, att = xb[i], atts[i]
                sent = [ID_token[id] for id in x]
                sent = ' '.join(sent)
                sentences.append(sent)
                attentions.append(att)

        pkl_path = 'att.pkl'
        f = open(pkl_path, 'wb')
        cPickle.dump([sentences, attentions], f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()
        print 'Dump attention to ' + pkl_path


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
    parser.add_argument('-att', type=int, default=0)
    parser.add_argument('-sq', type=int, default=0)
    parser.add_argument('-fine', type=int, default=0)
    args = parser.parse_args()

    args.dec = bool(args.dec)
    args.share = bool(args.share)
    args.att = bool(args.att)
    args.sq = bool(args.sq)
    args.fine = bool(args.fine)

    activation = None
    if args.model == 'dan':
        activation = 'tanh'
    elif args.model == 'tagm':
        activation = 'relu'

    num_epoch = 1
    if args.model == 'dan':
        num_epoch = 50

    cross_validation(drop=args.drop, hidden_dim=args.hid, dec=args.dec, update=args.update, lamb=args.lamb,
                     model=args.model, share=args.share, activation=activation, num_epoch=num_epoch,
                     need_attention=args.att, sq_loss=args.sq, fine=args.fine)


if __name__ == '__main__':
    test1()
