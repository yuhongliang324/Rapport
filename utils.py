__author__ = 'yuhongliang324'

import os
from collections import defaultdict
from scipy.io import loadmat
import numpy
import math
from data_preprocessing.krippendorff_alpha import krippendorff_alpha as ka
from data_path import info_root


def rename(root):
    files = os.listdir(root)
    for fn in files:
        if not fn.startswith('D'):
            continue
        pathname = os.path.join(root, fn)
        if not os.path.isdir(pathname):
            continue
        files1 = os.listdir(pathname)
        for fn1 in files1:
            if not fn1.endswith('mat'):
                continue
            sp = fn1.split('_')
            if not sp[0].startswith('D'):
                sp = sp[1:]
            dyad_name, session_name = sp[0], sp[1]
            slice_id = sp[3]
            if slice_id.startswith('Slice'):
                slice_id = slice_id[5:]
            slice_id = slice_id.zfill(3)
            new_name = dyad_name + '_' + session_name + '_' + slice_id + '_' + sp[-1]
            os.rename(os.path.join(pathname, fn1), os.path.join(pathname, new_name))


# The main function to get the slice ratings
def get_slice_ratings(rating_root, outfile):
    # The function called by the main slice rating function
    def get_slice_ratings2(rating_csv):
        print rating_csv
        reader = open(rating_csv)
        lines = reader.readlines()
        reader.close()
        lines = lines[0].split('\r')[1:]
        lines = map(lambda x: x.strip(), lines)

        ret = []
        for line in lines:
            sp = line.split(',')
            if len(sp[7]) == 0:
                continue
            rating = float(sp[7])
            dyad, session, slice = int(sp[0]), int(sp[1]), int(sp[2])
            ret.append((dyad, session, slice, rating))
        return ret
    files = os.listdir(rating_root)
    slice_rating = defaultdict(float)
    slice_num = defaultdict(int)
    for fn in files:
        if not fn.endswith('csv'):
            continue
        ret = get_slice_ratings2(os.path.join(rating_root, fn))
        for dyad, session, slice, rating in ret:
            slice_name = str(dyad) + ',' + str(session) + ',' + str(slice).zfill(3)
            slice_num[slice_name] += 1
            slice_rating[slice_name] = \
                (slice_rating[slice_name] * (slice_num[slice_name] - 1) + rating)/slice_num[slice_name]
    slice_rating_sorted = sorted(slice_rating.iteritems(), key=lambda d: d[0])
    writer = open(outfile, 'w')
    for slice, rating in slice_rating_sorted:
        writer.write(slice + ',' + str(rating) + '\n')
    writer.close()


# side: l - left only, r - right only, lr - left and right, b - concatenation of lr, ba - adding of lr
def load_feature_vision(mat_file, feature_name='hog', side='ba', only_suc=True):
    lfeat_name = 'left_' + feature_name + '_feature'
    rfeat_name = 'right_' + feature_name + '_feature'
    data = loadmat(mat_file)
    lfeat, rfeat = data[lfeat_name], data[rfeat_name]
    lsuc, rsuc = numpy.squeeze(data['left_success']), numpy.squeeze(data['right_success'])
    rating = float(data['label'][0])
    if side == 'l':
        if lfeat.dtype == numpy.int16:
            return None
        if only_suc:
            lfeat = lfeat[lsuc == 1]
            return lfeat, rating
        return lfeat, lsuc, rating
    elif side == 'r':
        if rfeat.dtype == numpy.int16:
            return None
        if only_suc:
            rfeat = rfeat[rsuc == 1]
            return rfeat, rating
        return rfeat, rsuc, rating
    elif side == 'lr':
        if lfeat.dtype == numpy.int16 or rfeat.dtype == numpy.int16:
            return None
        if only_suc:
            lfeat = lfeat[lsuc == 1]
            rfeat = rfeat[rsuc == 1]
            return (lfeat, rfeat), rating
        return (lfeat, rfeat), (lsuc, rsuc), rating
    else:
        if lfeat.dtype == numpy.int16 or rfeat.dtype == numpy.int16:
            return None
        if 'a' in side:  # Use add for both sides
            feat = lfeat + rfeat
        else:
            feat = numpy.concatenate((lfeat, rfeat), axis=1)
        suc = numpy.logical_and(lsuc == 1, rsuc == 1)
        if only_suc:
            feat = feat[suc]
            return feat, rating
        return feat, suc, rating


def get_ratings(rating_root=info_root):
    slice_ratings = {}

    def get_ratings1(rating_csv):
        print rating_csv
        reader = open(rating_csv)
        lines = reader.readlines()
        reader.close()
        lines = lines[0].split('\r')
        lines = map(lambda x: x.strip(), lines)

        num_lines = len(lines)
        sp = lines[0].split(',')
        raters = sp[11:]

        for i in xrange(1, num_lines):
            line = lines[i]
            sp = line.split(',')
            if len(sp[7]) == 0:
                continue

            dyad, session, slice = int(sp[0]), int(sp[1]), int(sp[2])
            SliceName = str(dyad) + '_' + str(session) + '_' + str(slice)

            str_ratings = sp[11:]

            if SliceName in slice_ratings:
                rater_rating = slice_ratings[SliceName]
            else:
                rater_rating = {}
            for j, r in enumerate(str_ratings):
                if r.isdigit():
                    rater_rating[raters[j]] = float(r)
            num_inval = 0
            for rater, rating in rater_rating.items():
                if rating == 0:
                    num_inval += 1
            if num_inval >= 2:
                continue

            slice_ratings[SliceName] = rater_rating

    files = os.listdir(rating_root)
    for fn in files:
        if not fn.endswith('csv'):
            continue
        get_ratings1(os.path.join(rating_root, fn))
    return slice_ratings


def get_rater_agreement(rating_root, self_included=False, num_coders=4):
    slice_ratings = get_ratings(rating_root)

    # get RMSE
    rmse = 0.
    total = 0
    dyads = {'3', '4', '5', '6', '7'}
    dyad_rmse = defaultdict(float)
    dyad_total = defaultdict(int)
    for slice, rr in slice_ratings.items():
        if slice[0] not in dyads:
            continue
        ratings = rr.values()
        s = sum(ratings)
        if self_included:
            avg = s / float(len(ratings))
        for r in ratings:
            if not self_included:
                avg = (s - r) / float(len(ratings) - 1)
            dyad_rmse[slice[0]] += (r - avg) * (r - avg)
            dyad_total[slice[0]] += 1
    print 'Dyads',
    for d in dyads:
        dyad_rmse[d] = math.sqrt(dyad_rmse[d] / dyad_total[d])
        print d + ':' + str(dyad_rmse[d]),
    print
    avg_rmse = 0.
    for d in dyads:
        avg_rmse += dyad_rmse[d]
    avg_rmse /= len(dyad_rmse)
    print 'Average RMSE = %f' % avg_rmse

    for rr in slice_ratings.values():
        ratings = rr.values()
        s = sum(ratings)
        if self_included:
            avg = s / float(len(ratings))
        for r in ratings:
            if not self_included:
                avg = (s - r) / float(len(ratings) - 1)
            rmse += (r - avg) * (r - avg)
            total += 1
    rmse = math.sqrt(rmse / total)
    print 'RMSE = %f' % rmse

    # get Krip Alpha
    coders = [[] for _ in xrange(num_coders)]
    missing = '*'
    for rr in slice_ratings.values():
        ratings = rr.values()
        size = min(len(ratings), num_coders)
        for i in xrange(size):
            coders[i].append(str(ratings[i]))
        if size < num_coders:
            for i in xrange(size, num_coders):
                coders[i].append(missing)
    for i in xrange(num_coders):
        print len(coders[i]),
    print
    print 'Calculating Krip Alpha'
    print ka(coders, missing_items=missing)


data_root = '/multicomp/users/liangke/RAPT/features'


def test1():
    rename('/multicomp/users/liangke/RAPT/separate_audio_data/')


def test2():
    get_slice_ratings(info_root, os.path.join(info_root, 'ratings.txt'))


def test3():
    get_rater_agreement(info_root, self_included=True)


if __name__ == '__main__':
    test3()
