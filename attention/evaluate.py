__author__ = 'yuhongliang324'
import sys
sys.path.append('..')
from utils import get_ratings, get_coder
import math
from ensemble import combine


def load_gt():
    slice_ratings = get_ratings()
    print set(slice_ratings.keys())


def get_krip_alpha(slice_ratings):
    ratings = slice_ratings.values()
    ratings = [r.values() for r in ratings]
    ratings = [r for r in ratings if len(r) > 1]

    n_pairable = sum(len(r) for r in ratings)
    D_o = 0.
    size = len(ratings)
    for cnt, rs in enumerate(ratings):
        if cnt % 1000 == 0:
            print 'Stage 1', cnt, '/', size
        m_u = len(rs)
        tmp = 0.
        for i in xrange(m_u):
            for j in xrange(m_u):
                tmp += (rs[i] - rs[j]) * (rs[i] - rs[j])
        tmp /= float(m_u) - 1
        D_o += tmp
    D_o /= n_pairable

    all_ratings = []
    for rs in ratings:
        for r in rs:
            all_ratings.append(r)
    D_e = 0.
    size = len(all_ratings)
    for i in xrange(size):
        if i % 1000 == 0:
            print 'Stage 2', i, '/', size
        for j in xrange(size):
            D_e += (all_ratings[i] - all_ratings[j]) * (all_ratings[i] - all_ratings[j])
    D_e /= n_pairable * (n_pairable - 1)

    alpha = 1. - D_o / D_e
    print alpha
    return alpha


def get_krip_alpha_given_coder(slice_ratings, coder):
    '''
    :param slice_ratings: dict. Item example: {'5_1_50': {'coder1': 7.0, 'coder2': 6.0, 'coder3': 5.0, 'coder4': 7.0}}
    :param coder: dict. Item example: {'5_1_50': 5.32}
    :return: double
    '''
    pred = []
    refereces = []
    for slice, rating in coder.items():
        pred.append(rating)
        refereces.append(slice_ratings[slice].values())

    D_o = 0.
    size = len(pred)
    for i in xrange(size):
        tmp = 0.
        for rf in refereces[i]:
            tmp += (pred[i] - rf) * (pred[i] - rf)
        tmp /= len(refereces[i])
        D_o += tmp
    D_o /= size

    D_e = 0.
    all_refs = []
    for rs in refereces:
        for r in rs:
            all_refs.append(r)
    for cnt, pr in enumerate(pred):
        if cnt % 200 == 0:
            print cnt, '/', size
        for r in all_refs:
            D_e += (pr - r) * (pr - r)
    D_e /= len(all_refs) * len(pred)

    alpha = 1. - D_o / D_e
    return alpha


def get_rmse(slice_ratings, coder):
    rmse = 0.
    rmse_skyline = 0.
    total = 0
    for slice, rating in coder.items():
        refs = slice_ratings[slice].values()
        sum_refs = sum(refs)
        len_refs = len(refs)
        gt = sum_refs / float(len_refs)
        rmse += (rating - gt) * (rating - gt)
        for rf in refs:
            gt = (sum_refs - rf) / (float(len_refs) - 1.)
            rmse_skyline += (rf - gt) * (rf - gt)
        total += len_refs
    rmse /= len(coder)
    rmse = math.sqrt(rmse)

    rmse_skyline /= total
    rmse_skyline = math.sqrt(rmse_skyline)

    return rmse, rmse_skyline


def test1():
    load_gt()


def test2():
    slice_ratings = get_ratings()
    get_krip_alpha(slice_ratings)


def test3():
    slice_ratings = get_ratings()
    coder = get_coder('../results/result_audio_b_drop_0.25_w_0.5_fact_None.txt')
    alpha = get_krip_alpha_given_coder(slice_ratings, coder)
    rmse, rmse_skyline = get_rmse(slice_ratings, coder)
    print 'alpha = %f, rmse = %f, rmse_skyline = %f' % (alpha, rmse, rmse_skyline)


def test4():
    slice_ratings = get_ratings()
    coder1 = get_coder('../results/result_hog_ba.txt')
    coder2 = get_coder('../results/result_audio_b_drop_0.25_w_0.25_fact_None.txt')
    coder = combine([coder1, coder2])
    alpha = get_krip_alpha_given_coder(slice_ratings, coder)
    rmse, rmse_skyline = get_rmse(slice_ratings, coder)
    print 'alpha = %f, rmse = %f, rmse_skyline = %f' % (alpha, rmse, rmse_skyline)


if __name__ == '__main__':
    test4()

