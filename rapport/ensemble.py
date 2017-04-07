__author__ = 'yuhongliang324'


def combine(coders, weights=None):
    def all_have(slice):
        for coder in coders:
            if slice not in coder:
                return False
        return True

    num_coder = len(coders)
    if weights is None:
        weights = [1. / num_coder for _ in xrange(num_coder)]
    slice_rating = {}
    for i in xrange(num_coder):
        for slice, rating in coders[i].items():
            if not all_have(slice):
                continue
            if slice in slice_rating:
                slice_rating[slice] += rating * weights[i]
            else:
                slice_rating[slice] = rating * weights[i]
    return slice_rating
