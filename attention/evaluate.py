__author__ = 'yuhongliang324'
import sys
sys.path.append('..')
from utils import get_ratings


def load_gt():
    slice_ratings = get_ratings()
    print set(slice_ratings.keys())


def get_krip_alpha(slice_ratings):
    ratings = slice_ratings.values()
    ratings = [r.values() for r in ratings]
    ratings = [r for r in ratings if len(r) > 1]
    D_o = 0.
    for rs in ratings:
        size = len(rs)
        for i in xrange(size):
            for j in xrange(size):



if __name__ == '__main__':
    load_gt()
