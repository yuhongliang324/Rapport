__author__ = 'yuhongliang324'

import cPickle
from collections import defaultdict


def extract_top_words():
    reader = open('att_tmp.pkl')
    sentences, attentions = cPickle.load(reader)
    reader.close()
    word_weight = defaultdict(float)
    word_count = defaultdict(float)
    for sent, att in zip(sentences, attentions):
        words = sent.split()
        length = len(words)
        att *= length
        for word, aw in zip(words, att):
            word_weight[word] += aw
            word_count[word] += 1.
    for word, weight in word_weight.items():
        if word_count[word] < 20:
            word_weight[word] = 0.
        else:
            word_weight[word] = weight / word_count[word]
    word_weight_list = sorted(word_weight.iteritems(), key=lambda d: d[1], reverse=True)
    for i in xrange(50):
        word, weight = word_weight_list[i][0], word_weight_list[i][1]
        print word.replace('.', '') + '\t' + str(weight)


def test1():
    extract_top_words()


if __name__ == '__main__':
    test1()
