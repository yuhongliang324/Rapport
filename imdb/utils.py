__author__ = 'yuhongliang324'
import os
from nltk import word_tokenize


origin_data_root = '/usr0/home/hongliay/datasets/aclImdb/'
origin_train_pos_root = os.path.join(origin_data_root, 'train/pos/')
origin_train_neg_root = os.path.join(origin_data_root, 'train/neg/')
origin_test_pos_root = os.path.join(origin_data_root, 'test/pos/')
origin_test_neg_root = os.path.join(origin_data_root, 'test/neg/')

POSITIVE = 0
NEGATIVE = 1

data_root = os.path.join(origin_data_root, 'processed_data/')
train_file = os.path.join(data_root, 'train.txt')
test_file = os.path.join(data_root, 'test.txt')


def process_to_single_file(pos_path, neg_path, out_file):
    writer = open(out_file, 'w')

    def write_file(dir_path, label):
        files = os.listdir(dir_path)
        num_files = len(files)
        files.sort()
        for i, fn in enumerate(files):
            if not fn.endswith('txt'):
                continue
            if (i + 1) % 1000 == 0:
                print i + 1, '/', num_files
            fp = os.path.join(dir_path, fn)
            reader = open(fp)
            lines = reader.readlines()
            reader.close()
            line = lines[0].strip().lower()
            line = line.replace('<br />', ' ')
            tokens = word_tokenize(line)
            line = ' '.join(tokens)
            writer.write(str(label) + ' ' + line + '\n')

    write_file(pos_path, POSITIVE)
    write_file(neg_path, NEGATIVE)
    writer.close()


def test1():
    process_to_single_file(origin_train_pos_root, origin_train_neg_root, train_file)
    process_to_single_file(origin_test_pos_root, origin_test_neg_root, test_file)


if __name__ == '__main__':
    test1()
