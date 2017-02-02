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
        plt.figure(figsize=(20, 6))
        # plt.plot(mins, '--', label='Min Annotation')
        # plt.plot(maxs, '--', label='Max Annotation')
        x = numpy.arange(len(mins))
        plt.errorbar(x, avgs, yerr=[avgs - mins, maxs - avgs], fmt='c-', label='Annotations')
        # plt.plot(avgs, label='Average Annotation')
        plt.plot(pred, label='Predictions', color='b', linewidth=2.)
        plt.legend()
        plt.title('Dyad ' + str(dyad))
        plt.savefig(os.path.join(img_root, 'dyad_' + str(dyad)))


def test1():
    slice_ratings = get_ratings()
    message = 'result_audio_b_drop_0.25_w_0.0_fact_None'
    coder = get_coder('../results/' + message + '.txt')
    visualize(slice_ratings, coder, '../predictions/' + message)


if __name__ == '__main__':
    test1()

