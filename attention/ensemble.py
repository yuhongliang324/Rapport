__author__ = 'yuhongliang324'


def combine(coders):
    num_coder = len(coders)
    slice_rating = {}
    slice_count = {}
    for i in xrange(num_coder):
        for slice, rating in coders[i].items():
            if slice in slice_rating:
                slice_rating[slice] += rating
                slice_count[slice] += 1.
            else:
                slice_rating[slice] = rating
                slice_count[slice] = 1.
    for slice, rating in slice_rating.items():
        slice_rating[slice] = rating / slice_count[slice]
    return slice_rating
