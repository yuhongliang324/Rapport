__author__ = 'yuhongliang324'

import os


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
            if not (fn1.endswith('txt') or fn.endswith('hog')):
                continue
            sp = fn1.split('_')
            dyad_name, session_name = sp[1], sp[2]
            slice_id = sp[4][5:].zfill(3)
            new_name = dyad_name + '_' + session_name + '_' + slice_id + '_' + sp[-1]
            print fn1, new_name

data_root = '/multicomp/users/liangke/RAPT/features'

rename(data_root)
