__author__ = 'yuhongliang324'
import numpy
from sklearn.preprocessing import normalize
from utils import load_feature
import os

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


def calculate_similarity(mat_file, feature_name='hog', cosine=True, topK=10):
    lfeat, rfeat, lsuc, rsuc = load_feature(mat_file, feature_name=feature_name, side='lr', only_suc=False)
    if cosine:
        lfeat = normalize(lfeat)
        rfeat = normalize(rfeat)
    sim = numpy.dot(lfeat, rfeat.T)
    top_sim_pairs = []
    top_dissim_pairs = []
    sim_index = sim.argsort(axis=None)
    count = 0
    for i in xrange(sim_index.shape[0]):
        ind = sim_index[i]
        lindex = ind // sim.shape[1]
        rindex = ind % sim.shape[1]
        if not (lsuc[lindex] == 1 and rsuc[rindex] == 1):
            continue
        top_dissim_pairs.append((lindex, rindex))
        count += 1
        if count >= topK:
            break
    sim_index = sim_index[::-1]
    count = 0
    for i in xrange(sim_index.shape[0]):
        ind = sim_index[i]
        lindex = ind // sim.shape[1]
        rindex = ind % sim.shape[1]
        if not (lsuc[lindex] == 1 and rsuc[rindex] == 1):
            continue
        top_sim_pairs.append((lindex, rindex))
        count += 1
        if count >= topK:
            break
    return top_sim_pairs, top_dissim_pairs


def find_all_similarity(sim_file, dissim_file, feature_name='hog', topK=10):
    from data_path import sample_10_root

    writer_sim = open(sim_file, 'w')
    writer_dissim = open(dissim_file, 'w')

    dyads = os.listdir(sample_10_root)
    dyads.sort()
    for dyad_name in dyads:
        print dyad_name
        dyad_path = os.path.join(sample_10_root, dyad_name)
        if not os.path.isdir(dyad_path):
            continue
        slices = os.listdir(dyad_path)
        slices.sort()
        for slice_name in slices:
            if not slice_name.endswith('mat'):
                continue
            slice_mat = os.path.join(dyad_path, slice_name)
            top_sim_pairs, top_dissim_pairs = calculate_similarity(slice_mat, feature_name=feature_name, topK=topK)
            sp = slice_name[:-4].split('_')
            dyad_ID, session_ID, slice_ID = sp[0][1:], sp[1][1:], str(int(sp[3]))
            for i in xrange(topK):
                l, r = str(top_sim_pairs[i][0] * 10), str(top_sim_pairs[i][1] * 10)
                writer_sim.write('sim\t' + dyad_ID + ',' + session_ID + ',' + slice_ID + ',' + l + ',' + r + '\n')
                l, r = str(top_dissim_pairs[i][0] * 10), str(top_dissim_pairs[i][1] * 10)
                writer_dissim.write('dissim\t' + dyad_ID + ',' + session_ID + ',' + slice_ID + ',' + l + ',' + r + '\n')
    writer_sim.close()
    writer_dissim.close()


def test1():
    get_high_rating_slices('../data_info/AMT_Batch1_results_large.csv')


def test2():
    feature_name = 'hog'
    topK = 10
    find_all_similarity('../tmp/sim.txt', '../tmp/dissim.txt', feature_name=feature_name, topK=topK)


if __name__ == '__main__':
    test2()


