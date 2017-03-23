__author__ = 'yuhongliang324'


Emotions = ['Angry', 'Happy', 'Sad', 'Neutral']


def evaluate(fn):
    reader = open(fn)
    lines = reader.readlines()
    reader.close()
    lines = map(lambda x: x.strip(), lines)
    right = [0, 0, 0, 0]
    total = [0, 0, 0, 0]
    for line in lines:
        sp = line.split(',')
        pred, actual = int(sp[0]), int(sp[1])
        total[pred] += 1
        if pred == actual:
            right[pred] += 1
    right4 = sum(right)
    total4 = sum(total)
    for i in xrange(len(right)):
        right /= float(total)
    for i in xrange(len(right)):
        print Emotions[i], right[i], '\t'
    print right4 / float(total4)
