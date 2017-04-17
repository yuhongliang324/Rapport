__author__ = 'yuhongliang324'

import cPickle
from collections import defaultdict


def extract_top_words():
    reader = open('att.pkl')
    sentences, attentions = cPickle.load(reader)
    reader.close()
    word_weight = defaultdict(float)
    word_count = defaultdict(float)
    for sent, att in zip(sentences, attentions):
        words = sent.split()
        length = len(words)
        att *= length
        for word, aw in zip(words, att):
            word_weight += aw
            word_count += 1.
    for word, weight in word_weight.items():
        word_weight[word] = weight / word_count[word]
    word_weight_list = sorted(word_weight.iteritems(), key=lambda d: d[1], reverse=True)
    for i in xrange(100):
        word, weight = word_weight_list[i][0], word_weight_list[i][1]
        print word, weight
