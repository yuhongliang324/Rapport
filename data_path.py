__author__ = 'yuhongliang324'
import os

data_root = '/multicomp/users/liangke/RAPT/'
sample_10_root = os.path.join(data_root, 'sampled_features_10')
sample_30_root = os.path.join(data_root, 'sampled_features_30')
log_root = '../log/'

audio_root = os.path.join(data_root, 'separate_audio_data')

info_root = '../data_info/'
ratings_file = os.path.join(info_root, 'ratings.txt')
ratings_best3_file = os.path.join(info_root, 'ratings_best3.txt')
