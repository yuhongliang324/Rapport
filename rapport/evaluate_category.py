__author__ = 'yuhongliang324'


def get_accuracy(result_file):
    reader = open(result_file)
    lines = reader.readlines()
    reader.close()
    total = 0
    right = 0
    count = [0, 0, 0]
    for line in lines:
        sp = line.strip().split(',')
        pred, gt = int(sp[-2]), int(sp[-1])
        if pred == gt:
            right += 1
        total += 1
        count[gt] += 1.
    acc = float(right) / total
    mj_acc = max(count) / float(total)
    print 'Accuracy = %f' % acc
    print 'Majority Accuracy = %f' % mj_acc
    print count[0] / total, count[1] / total, count[2] / total


def test1():
    result_file = '../results/result_ad_hog_lr_model_gru_share_False_drop_0.0_lamb_0.0_fact_None_cat.txt'
    get_accuracy(result_file)


if __name__ == '__main__':
    test1()
