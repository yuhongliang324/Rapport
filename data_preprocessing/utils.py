__author__ = 'yuhongliang324'

import os
from collections import defaultdict

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
            if not (fn1.endswith('txt') or fn1.endswith('hog')):
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



data_root = '/multicomp/users/liangke/RAPT/features'
data_info_root = '../data_info/'


get_slice_ratings2(data_info_root, os.path.join(data_info_root, 'ratings.txt'))