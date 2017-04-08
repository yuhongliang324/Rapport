__author__ = 'yuhongliang324'


def combine(coders, weights=None):
    num_coder = len(coders)

    def all_coef(slice):
        z = 0.
        for i in xrange(num_coder):
            coder = coders[i]
            if slice in coder:
                z += weights[i]
        return z

    if weights is None:
        weights = [1. / num_coder for _ in xrange(num_coder)]
    slice_rating = {}
    for i in xrange(num_coder):
        for slice, rating in coders[i].items():
            z = all_coef(slice)
            if slice in slice_rating:
                slice_rating[slice] += rating * weights[i] / z
            else:
                slice_rating[slice] = rating * weights[i] / z
    return slice_rating
