__author__ = 'yuhongliang324'
import operator
import numpy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import shutil
import sys
sys.path.append('..')
from utils import get_ratings, get_coder
from ensemble import combine


def visualize(ground_truth, coder, img_root):
    sorted_gt = {}
    for slice_str, ratings in ground_truth.items():
        sp = slice_str.split('_')
        dyad, session, slice = sp[0].zfill(2), sp[1], sp[2].zfill(3)
        sorted_gt[dyad + '_' + session + '_' + slice] = sorted(ratings.values())
    sorted_gt = sorted(sorted_gt.items(), key=operator.itemgetter(0))
    dyad_min, dyad_max, dyad_avg = {}, {}, {}
    dyad_pred = {}

    for slice_str, ratings in sorted_gt:
        sp = slice_str.split('_')
        dyad, session, slice = int(sp[0]), int(sp[1]), int(sp[2])
        if dyad not in dyad_min:
            dyad_min[dyad] = []
            dyad_max[dyad] = []
            dyad_avg[dyad] = []
            dyad_pred[dyad] = []
        dyad_min[dyad].append(ratings[0])
        dyad_max[dyad].append(ratings[-1])
        s = sum(ratings)
        avg = float(s) / len(ratings)
        dyad_avg[dyad].append(avg)
        slice_str2 = str(dyad) + '_' + str(session) + '_' + str(slice)
        if slice_str2 in coder:
            dyad_pred[dyad].append(coder[slice_str2])
        else:
            dyad_pred[dyad].append(numpy.nan)

    if os.path.isdir(img_root):
        shutil.rmtree(img_root)
    os.mkdir(img_root)

    for dyad in dyad_pred.keys():
        mins, maxs, avgs = dyad_min[dyad], dyad_max[dyad], dyad_avg[dyad]
        mins = numpy.asarray(mins)
        maxs = numpy.asarray(maxs)
        avgs = numpy.asarray(avgs)
        pred = dyad_pred[dyad]
        pred = numpy.asarray(pred)
        if pred[numpy.isnan(pred)].shape[0] == pred.shape[0]:
            continue
        print 'Dyad', dyad
        plt.figure(figsize=(20, 6))
        plt.ylim([0, 7])
        # plt.plot(mins, '--', label='Min Annotation')
        plt.plot(avgs, color='c', label='Ground Truth')
        x = numpy.arange(len(mins))
        # plt.errorbar(x, avgs, yerr=[avgs - mins, maxs - avgs], fmt='c-', label='Annotations')
        # plt.plot(avgs, label='Average Annotation')
        plt.plot(pred, label='Predictions', color='b', linewidth=2.)
        plt.legend()
        plt.title('Dyad ' + str(dyad))
        plt.savefig(os.path.join(img_root, 'dyad_' + str(dyad)))


def test1():
    slice_ratings = get_ratings()
    message = 'result_attention_only_audio_b_drop_0.0_fact_None'
    coder = get_coder('../results/' + message + '.txt')
    visualize(slice_ratings, coder, '../predictions/' + message)


def test2():
    slice_ratings = get_ratings()
    message = 'svr_result_1'
    coder = get_coder('../results/' + message + '.txt')
    visualize(slice_ratings, coder, '../predictions/' + message)


def test3():
    slice_ratings = get_ratings()
    coder1 = get_coder('../results/result_audio_b_drop_0.1_w_0.0_fact_None.txt')
    coder2 = get_coder('../results/svr_result.txt')
    coder = combine([coder1, coder2])
    visualize(slice_ratings, coder, '../predictions/' + 'rnn+svr')


if __name__ == '__main__':
    test1()

