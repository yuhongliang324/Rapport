__author__ = 'yuhongliang324'
import sys
sys.path.append('..')
from utils import get_ratings


def load_gt():
    slice_ratings = get_ratings()
    print set(slice_ratings.keys())


if __name__ == '__main__':
    load_gt()
