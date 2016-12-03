__author__ = 'yuhongliang324'


def get_high_rating_slices(rating_csv, threshold=5):
    reader = open(rating_csv)
    lines = reader.readlines()
    reader.close()
    lines = lines[0].split('\r')[1:]
    lines = map(lambda x: x.strip(), lines)

    ret = []

    for line in lines:
        sp = line.split(',')
        rating = float(sp[7])
        if rating < threshold:
            continue
        dyad, session, slice = int(sp[0]), int(sp[1]), int(sp[2])
        ret.append((dyad, session, slice))
    return ret


get_high_rating_slices('../data_info/AMT_Batch1_results_large.csv')