__author__ = 'yuhongliang324'

for d in xrange(1, 1008):
    if ((2017 - d) * d) % (2016 - 2 * d) == 0:
        m = (2017 - d) * d / (2016 - 2 * d)
        print d, m - d, m

N = 100
happy = [False for i in xrange(N)]
visited = [False for j in xrange(N)]
last_visit = []
while True:
