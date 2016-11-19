__author__ = 'yuhongliang324'

import misvm
import numpy
from scipy.io import loadmat
import random
from data_path import *


class MIL:

    def __init__(self, root, LR='both', binarize=True, feat_type='hog', kernel='linear', C=0.1, logfile=None):
        self.LR = LR
        self.binarize = binarize
        self.feat_type = feat_type
        self.kernel, self.C = kernel, C
        self.logfile = logfile

        self.features = []
        self.labels = []
        self.dname = []

        files = os.listdir(root)
        for fn in files:
            dirname = os.path.join(root, fn)
            if not os.path.isdir(dirname):
                continue
            self.dname.append(fn)
            self.add_dyad_data(dirname)
        self.num_dyad = len(self.features)

        self.classifier = misvm.MISVM(kernel=self.kernel, C=self.C, max_iters=20)

    def add_dyad_data(self, dirname):
        dfeatures = []
        dlabels = []
        files = os.listdir(dirname)
        for fn in files:
            fn_path = os.path.join(dirname, fn)
            if not fn_path.endswith('.mat'):
                continue
            data = loadmat(fn_path)

            if self.LR == 'L' or self.LR == 'l' or self.LR == 'Left' or self.LR == 'left':
                suc = numpy.squeeze(data['left_success'])
                if self.feat_type == 'gemo':
                    f = data['left_gemo_feature']
                else:
                    f = data['left_hog_feature']
            elif self.LR == 'R' or self.LR == 'r' or self.LR == 'Right' or self.LR == 'right':
                suc = numpy.squeeze(data['right_success'])
                if self.feat_type == 'gemo':
                    f = data['right_gemo_feature']
                else:
                    f = data['right_hog_feature']
            else:
                suc = numpy.squeeze(data['left_success']) * numpy.squeeze(data['right_success'])
                if self.feat_type == 'gemo':
                    f = numpy.concatenate((data['left_gemo_feature'], data['right_gemo_feature']), axis=1)
                else:
                    f = numpy.concatenate((data['left_hog_feature'], data['right_hog_feature']), axis=1)
            f = f[suc == 1]
            label = data['label'][0, 0]
            if f.shape[0] == 0 or label == 0:
                continue
            if self.binarize:
                if label >= 4:
                    label = 1
                else:
                    label = -1
            dfeatures.append(f)
            dlabels.append(label)

        self.features.append(dfeatures)
        self.labels.append(dlabels)

    def validate(self):
        acc = 0.
        maj = 0.
        self.writer = open(self.logfile, 'w')
        for i in xrange(self.num_dyad):
            ret = self.leave_one_out(i)
            if self.binarize:
                acc += ret[0]
                maj += ret[1]
        acc /= self.num_dyad
        maj /= self.num_dyad

        self.writer.write('Total: Accuracy = %f, Majority = %f\n' % (acc, maj))
        self.writer.close()

    def leave_one_out(self, out_index):
        train_bags1, train_labels1 = [], []
        test_bags = self.features[out_index]
        test_labels = numpy.asarray(self.labels[out_index])

        for i in xrange(self.num_dyad):
            if i == out_index:
                continue
            size = len(self.features[i])
            for j in xrange(size):
                train_bags1.append(self.features[i][j])
                train_labels1.append(self.labels[i][j])

        size_train = len(train_bags1)
        ind_shuffle = range(size_train)
        random.shuffle(ind_shuffle)
        train_bags, train_labels = [], []
        for i in ind_shuffle:
            train_bags.append(train_bags1[i])
            train_labels.append(train_labels1[i])
        train_labels = numpy.array(train_labels)

        self.classifier.fit(train_bags, train_labels)

        pred = self.classifier.predict(test_bags)
        pred = numpy.sign(pred)

        dif = pred - test_labels
        correct = dif[dif == 0].shape[0]
        acc = float(correct) / float(dif.shape[0])
        self.writer.write(self.dname[out_index] + ' Accuracy = %f\n' % acc)
        if self.binarize:
            num_pos = test_labels[test_labels > 0].shape[0]
            num_neg = test_labels[test_labels < 0].shape[0]
            ratio_pos = float(num_pos) / test_labels.shape[0]
            ratio_neg = float(num_neg) / test_labels.shape[0]
            self.writer.write('Positive samples = %d (%f), Negative samples = %d (%f)\n'\
                  % (num_pos, ratio_pos, num_neg, ratio_neg))

            return acc, max(ratio_pos, ratio_neg)
        return acc


def test1():
    samp = 30
    LR = 'b'
    feat_type = 'gemo'
    kernel = 'linear'
    C = 0.03
    if samp == 0:
        root = sample_0_root
    elif samp == 10:
        root = sample_10_root
    elif samp == 20:
        root = sample_20_root
    else:
        root = sample_30_root

    logname = 'sample_' + str(samp) + '_' + LR + '_' + feat_type + '_' + kernel + '_C_' + str(C) + '.log'
    logfile = os.path.join(log_root, logname)
    mil = MIL(root, LR=LR, kernel=kernel, C=C, feat_type=feat_type, logfile=logfile)
    mil.validate()

if __name__ == '__main__':
    test1()
