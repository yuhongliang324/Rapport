__author__ = 'yuhongliang324'
import numpy
from scipy.stats import pearsonr


Emotions = ['Angry', 'Happy', 'Sad', 'Neutral']


def evaluate_acc(fn, num_class=4):
    reader = open(fn)
    lines = reader.readlines()
    reader.close()
    lines = map(lambda x: x.strip(), lines)
    right = [0 for _ in xrange(num_class)]
    total = [0 for _ in xrange(num_class)]
    for line in lines:
        sp = line.split(',')
        pred, actual = int(sp[0]), int(sp[1])
        total[pred] += 1
        if pred == actual:
            right[pred] += 1
    right4 = sum(right)
    total4 = sum(total)
    for i in xrange(len(right)):
        right[i] /= float(total[i])
    for i in xrange(len(right)):
        print Emotions[i], right[i]
    print right4 / float(total4)


def evaluate_reg(fn):
    data = numpy.loadtxt(fn, delimiter=',')
    x, y = data[:, 0], data[:, 1]
    mae = numpy.mean(numpy.abs(x - y))
    r = pearsonr(x, y)
    print 'MAE = %f\tPCC = %f' % (mae, r[0])


def test1():
    evaluate_acc('results/result_ad_video_model_lstm_share_False_lamb_0.0_drop_0.0_cat.txt')


def test2():
    evaluate_acc('../mosi/results/result_ad_openface_model_dan_share_True_lamb_0.0001_drop_0.5_cat.txt', num_class=2)


def test3():
    evaluate_reg('../mosi/results/result_ad_openface_model_dan_share_False_lamb_0.0001_drop_0.5.txt')


if __name__ == '__main__':
    test3()
