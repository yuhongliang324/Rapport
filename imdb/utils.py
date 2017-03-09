__author__ = 'yuhongliang324'
import os


origin_data_root = '/usr0/home/hongliay/datasets/aclImdb/'
origin_train_pos_root = os.path.join(origin_data_root, 'train/pos/')
origin_train_neg_root = os.path.join(origin_data_root, 'train/neg/')
origin_test_pos_root = os.path.join(origin_data_root, 'test/pos/')
origin_test_neg_root = os.path.join(origin_data_root, 'test/neg/')

data_root = os.path.join(origin_data_root, 'processed_data/')


def process_to_single_file(pos_path, neg_path, out_file):
    # writer = open(out_file, 'w')

    def write_file(dir_path):
        files = os.listdir(dir_path)
        files.sort()
        for fn in files:
            if not fn.endswith('txt'):
                continue
            fp = os.path.join(dir_path, fn)
            reader = open(fp)
            lines = reader.readlines()
            reader.close()
            if len(lines) == 1:
                print len(lines)
    write_file(pos_path)
    write_file(neg_path)
    # writer.close()


def test1():
    process_to_single_file(origin_train_pos_root, origin_train_neg_root, None)


if __name__ == '__main__':
    test1()
