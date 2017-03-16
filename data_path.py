__author__ = 'yuhongliang324'
import os

dn = os.path.dirname(os.path.abspath(__file__))

data_root = '/multicomp/users/liangke/RAPT/'
sample_10_root = os.path.join(data_root, 'sampled_features_10')
sample_30_root = os.path.join(data_root, 'sampled_features_30')
log_root = os.path.join(dn, 'log/')

audio_root = os.path.join(data_root, 'separate_audio_data')

info_root = os.path.join(dn, 'data_info/')
ratings_file = os.path.join(info_root, 'ratings.txt')
ratings_best3_file = os.path.join(info_root, 'ratings_best3.txt')
rating_class_file = os.path.join(info_root, 'rating_classes.txt')
rating_class_best3_file = os.path.join(info_root, 'rating_classes_best3.txt')
data_split_file = os.path.join(info_root, 'split.txt')
