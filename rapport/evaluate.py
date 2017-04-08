__author__ = 'yuhongliang324'
import sys
sys.path.append('..')
from utils import get_all_ratings, get_coder
import math
from ensemble import combine
from scipy.stats import pearsonr


def load_gt():
    slice_ratings = get_all_ratings()
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
        if slice not in slice_ratings:
            continue
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


def get_pearson_given_coder(slice_ratings, coder):
    '''
    :param slice_ratings: dict. Item example: {'5_1_50': {'coder1': 7.0, 'coder2': 6.0, 'coder3': 5.0, 'coder4': 7.0}}
    :param coder: dict. Item example: {'5_1_50': 5.32}
    :return: double
    '''
    pred = []
    refereces = []
    for slice, rating in coder.items():
        if slice not in slice_ratings:
            continue
        pred.append(rating)
        ref = float(sum(slice_ratings[slice].values())) / len(slice_ratings[slice].values())
        refereces.append(ref)
    r = pearsonr(pred, refereces)
    return r[0]


def get_rmse(slice_ratings, coder):
    rmse = 0.
    rmse_skyline = 0.
    total = 0
    for slice, rating in coder.items():
        if slice not in slice_ratings:
            continue
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


def get_mae(slice_ratings, coder):
    mae = 0.
    mae_skyline = 0.
    total = 0
    for slice, rating in coder.items():
        if slice not in slice_ratings:
            continue
        refs = slice_ratings[slice].values()
        sum_refs = sum(refs)
        len_refs = len(refs)
        gt = sum_refs / float(len_refs)
        mae += abs(rating - gt)
        for rf in refs:
            gt = (sum_refs - rf) / (float(len_refs) - 1.)
            mae_skyline += abs(rf - gt)
        total += len_refs
    mae /= len(coder)

    mae_skyline /= total

    return mae, mae_skyline


def get_acc(fn, num_class=3):
    reader = open(fn)
    lines = reader.readlines()
    reader.close()
    lines = map(lambda x: x.strip(), lines)
    right = [0 for _ in xrange(num_class)]
    total = [0 for _ in xrange(num_class)]
    for line in lines:
        sp = line.split(',')
        pred, actual = int(sp[-2]), int(sp[-1])
        total[pred] += 1
        if pred == actual:
            right[pred] += 1
    right4 = sum(right)
    total4 = sum(total)
    for i in xrange(len(right)):
        right[i] /= float(total[i]) + 0.001
    for i in xrange(len(right)):
        print right[i],
    print
    print right4 / float(total4)


def test1():
    load_gt()


def test2():
    slice_ratings = get_all_ratings()
    get_krip_alpha(slice_ratings)


def test3():
    slice_ratings = get_all_ratings(best3=False)
    coder = get_coder('../results/result_tagm_audio_b_share_False_drop_0.0_lamb_0.0_fact_None.txt')
    alpha = get_krip_alpha_given_coder(slice_ratings, coder)
    mae, mae_skyline = get_mae(slice_ratings, coder)
    r = get_pearson_given_coder(slice_ratings, coder)
    print 'mae = %f, mae_skyline = %f' % (mae, mae_skyline)
    print 'pearson =', r
    print 'alpha = %f' % alpha
    # alpha = get_krip_alpha(slice_ratings)
    # print 'alpha_skyline = %f' % alpha


def test3_1():
    get_acc('../results/result_ours_audio_b_share_False_drop_0.0_lamb_0.0_fact_None_cat.txt')


def test4():
    slice_ratings = get_all_ratings(best3=False)
    model = 'tagm'
    coder1 = get_coder('../results/result_' + model + '_audio_b_share_False_drop_0.0_lamb_0.0_fact_None.txt')
    coder2 = get_coder('../results/result_' + model + '_hog_lr_share_False_drop_0.0_lamb_0.0_fact_None.txt')
    ws = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]
    for w in ws:
        print 'Audio weight =', w
        coder = combine([coder1, coder2], [w, 1. - w])
        mae, mae_skyline = get_mae(slice_ratings, coder)
        r = get_pearson_given_coder(slice_ratings, coder)
        print 'mae = %f, mae_skyline = %f' % (mae, mae_skyline)
        print 'pearson =', r
        print


if __name__ == '__main__':
    test4()
    # evaluate('../results/audio_svr.txt')

