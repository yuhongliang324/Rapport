__author__ = 'yuhongliang324'

import os
from collections import defaultdict
from scipy.io import loadmat
import numpy


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
            if not fn1.endswith('avi'):
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


def get_slice_ratings(rating_csv):
    reader = open(rating_csv)
    lines = reader.readlines()
    reader.close()
    lines = lines[0].split('\r')[1:]
    lines = map(lambda x: x.strip(), lines)

    ret = []
    for line in lines:
        sp = line.split(',')
        rating = float(sp[7])
        dyad, session, slice = int(sp[0]), int(sp[1]), int(sp[2])
        ret.append((dyad, session, slice, rating))
    return ret


def get_slice_ratings2(rating_root, outfile):
    files = os.listdir(rating_root)
    slice_rating = defaultdict(float)
    slice_num = defaultdict(int)
    for fn in files:
        if not fn.endswith('csv'):
            continue
        ret = get_slice_ratings(os.path.join(rating_root, fn))
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


# side: l - left only, r - right only, lr - left and right, b - concatenation of lr
def load_feature(mat_file, feature_name='hog', side='lr', only_suc=True):
    lfeat_name = 'left_' + feature_name + '_feature'
    rfeat_name = 'right_' + feature_name + '_feature'
    data = loadmat(mat_file)
    lfeat, rfeat = data[lfeat_name], data[rfeat_name]
    lsuc, rsuc = numpy.squeeze(data['left_success']), numpy.squeeze(data['right_success'])
    rating = float(data['label'][0])
    if side == 'l':
        if only_suc:
            lfeat = lfeat[lsuc == 1]
            return lfeat, rating
        return lfeat, lsuc, rating
    elif side == 'r':
        if only_suc:
            rfeat = rfeat[rsuc == 1]
            return rfeat, rating
        return rfeat, lsuc, rating
    elif side == 'lr':
        if only_suc:
            lfeat = lfeat[lsuc == 1]
            rfeat = rfeat[rsuc == 1]
            return lfeat, rfeat, rating
        return lfeat, rfeat, lsuc, rsuc, rating
    else:
        print lfeat.shape, rfeat.shape
        feat = numpy.concatenate((lfeat, rfeat), axis=1)
        suc = numpy.logical_and(lsuc == 1, rsuc == 1)
        if only_suc:
            feat = feat[suc]
            return feat, rating
        return feat, suc, rating


data_root = '/multicomp/users/liangke/RAPT/features'
data_info_root = '../data_info/'


def test1():
    rename('/multicomp/users/liangke/RAPT/data/')


def test2():
    get_slice_ratings2(data_info_root, os.path.join(data_info_root, 'ratings.txt'))


if __name__ == '__main__':
    test1()
