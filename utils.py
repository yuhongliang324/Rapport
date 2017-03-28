__author__ = 'yuhongliang324'

import os
from collections import defaultdict
import numpy
import math
from data_preprocessing.krippendorff_alpha import krippendorff_alpha as ka
from data_path import info_root, ratings_best3_file, data_split_file, rating_class_file, rating_class_best3_file
from scipy.interpolate import interp1d
import random
from sklearn.preprocessing import StandardScaler


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
# Write ratings.txt
def write_slice_ratings(rating_root, outfile):
    # The function called by the main slice rating function
    def write_slice_ratings2(rating_csv):
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
        ret = write_slice_ratings2(os.path.join(rating_root, fn))
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


# Write ratings_best.txt
def write_slice_ratings_best3(out_file):
    slice_ratings = get_all_ratings(best3=True)
    slice_avgr = {}
    for slice, rater_rating in slice_ratings.items():
        avgr = sum(rater_rating.values()) / float(len(rater_rating.values()))
        slice_avgr[slice] = avgr
    slice_avgr_sorted = sorted(slice_avgr.iteritems(), key=lambda d: d[0])
    writer = open(out_file, 'w')
    for slice, avgr in slice_avgr_sorted:
        writer.write(slice.replace('_', ',') + ',' + str(avgr) + '\n')
    writer.close()


def write_slice_rating_classes(out_file, best3=True):
    slice_ratings = get_all_ratings(best3=best3, ignore0=False)
    rr_count = defaultdict(float)
    for rrs in slice_ratings.values():
        for rater, rating in rrs.items():
            rr = rater + '_' + str(int(rating))
            rr_count[rr] += 1
    content = []
    for slice, rrs in slice_ratings.items():
        weights = [0. for _ in xrange(3)]
        for rater, rating in rrs.items():
            rr = rater + '_' + str(int(rating))
            w = 1. / rr_count[rr]
            if rating < 4:
                weights[0] += w
            elif rating > 4:
                weights[2] += w
            else:
                weights[1] += w
        sp = slice.split('_')
        dyad, session, sliceID = sp[0], sp[1], sp[2].zfill(3)
        if weights[0] > weights[1] and weights[0] > weights[2]:
            cl = 0
        elif weights[2] > weights[0] and weights[2] > weights[1]:
            cl = 2
        else:
            cl = 1
        content.append(dyad + ',' + session + ',' + sliceID + ',' + str(cl))
    content.sort()
    writer = open(out_file, 'w')
    writer.write('\n'.join(content))
    writer.close()


def interpolate_features(X, suc):
    m = numpy.mean(suc)
    if m < 0.5:
        return None
    if m == 1.:
        return X
    ind = numpy.nonzero(suc)[0]
    max_ind = numpy.max(ind)
    min_ind = numpy.min(ind)
    X = X[ind]
    f = interp1d(ind, X, axis=0)
    ind = numpy.arange(suc.shape[0])
    ind[ind > max_ind] = max_ind
    ind[ind < min_ind] = min_ind
    return f(ind)


def get_all_ratings(rating_root=info_root, best3=True, ignore0=True):
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

            '''
            if SliceName in slice_ratings:
                rater_rating = slice_ratings[SliceName]
            else:
                rater_rating = {}'''

            # Cover directly:
            rater_rating = {}

            for j, r in enumerate(str_ratings):
                if r.isdigit():
                    rater_rating[raters[j]] = float(r)
            if ignore0:
                num_inval = 0
                for rater, rating in rater_rating.items():
                    if rating == 0:
                        num_inval += 1
                if num_inval >= 2:
                    continue

            slice_ratings[SliceName] = rater_rating

    files = os.listdir(rating_root)
    files.sort()
    for fn in files:
        if not fn.endswith('csv'):
            continue
        get_ratings1(os.path.join(rating_root, fn))
    if best3:
        slice_ratings = get_best3(slice_ratings)
    return slice_ratings


def get_best3(slice_ratings):
    slice_ratings_best3 = {}
    for slice, rater_rating in slice_ratings.items():
        nrater = len(rater_rating)
        if nrater <= 3:
            slice_ratings_best3[slice] = rater_rating
        else:
            tmp = list(rater_rating.items())
            min_var = 100000
            ir1, ir2, ir3 = None, None, None
            for i in xrange(nrater - 2):
                for j in xrange(i + 1, nrater - 1):
                    for k in xrange(j + 1, nrater):
                        ri, rj, rk = tmp[i][1], tmp[j][1], tmp[k][1]
                        var = numpy.var([ri, rj, rk])
                        if var < min_var:
                            min_var = var
                            ir1, ir2, ir3 = i, j, k
            rater_rating_best3 = {tmp[ir1][0]: tmp[ir1][1], tmp[ir2][0]: tmp[ir2][1], tmp[ir3][0]: tmp[ir3][1]}
            slice_ratings_best3[slice] = rater_rating_best3
    return slice_ratings_best3


def get_coder(file_name):
    reader = open(file_name)
    lines = reader.readlines()
    reader.close()
    lines = map(lambda x: x.strip(), lines)
    slice_rating = {}
    slice_count = {}
    for line in lines:
        sp = line.split(',')
        dyad, session, slice = sp[0], sp[1], sp[2]
        slice_str = dyad + '_' + session + '_' + slice
        rating = float(sp[3])
        if slice_str in slice_rating:
            slice_rating[slice_str] += rating
            slice_count[slice_str] += 1.
        else:
            slice_rating[slice_str] = rating
            slice_count[slice_str] = 1.
    for slice_str, rating in slice_rating.items():
        slice_rating[slice_str] = rating / slice_count[slice_str]
    return slice_rating


def get_rater_agreement(rating_root, self_included=False, num_coders=4):
    slice_ratings = get_all_ratings(rating_root)

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


def split_dataset(dyads, out_file):
    size = len(dyads)
    ind = range(size)
    while True:
        random.shuffle(ind)
        repeat = False
        for i in xrange(size):
            if i == ind[i]:
                repeat = True
                break
        if not repeat:
            break
    writer = open(out_file, 'w')
    for i in xrange(size):
        writer.write(str(dyads[i]) + ' ' + str(dyads[ind[i]]) + '\n')
    writer.close()


def load_split(split_file=data_split_file):
    reader = open(split_file)
    lines = reader.readlines()
    reader.close()
    lines = map(lambda x: x.strip(), lines)
    vals, tests = [], []
    for line in lines:
        sp = line.split()
        tests.append(int(sp[0]))
        vals.append(int(sp[1]))
    return vals, tests


# Input: a list of X arrays
def standardize(X_list):
    '''
    dim = X_list[0].shape[1]
    same = True
    for X in X_list:
        if X.shape[1] != dim:
            same = False
            break
    if not same:
        print [X.shape[1] for X in X_list]'''
    Xs = numpy.concatenate(X_list, axis=0)
    Xs_std = StandardScaler().fit_transform(Xs)

    X_list_std = []
    start = 0
    for X in X_list:
        end = start + X.shape[0]
        X_list_std.append(Xs_std[start: end])
        start = end
    return X_list_std


data_root = '/multicomp/users/liangke/RAPT/features'


def test1():
    rename('/multicomp/users/liangke/RAPT/separate_audio_data/')


def test2():
    write_slice_ratings(info_root, os.path.join(info_root, 'ratings.txt'))


def test3():
    get_rater_agreement(info_root, self_included=True)


def test4():
    write_slice_ratings_best3(ratings_best3_file)


def test5():
    dyads = [3, 4, 7, 8, 9, 10, 11, 12, 16, 17, 18]
    split_dataset(dyads, data_split_file)


def test6():
    write_slice_rating_classes(rating_class_file, best3=False)
    write_slice_rating_classes(rating_class_best3_file, best3=True)


if __name__ == '__main__':
    test5()
