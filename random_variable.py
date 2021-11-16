import numpy as np


class RandomVariable:
    def __init__(self, x, y, matrix):
        self.x = x
        self.y = y
        self.p = matrix
        self.q = list()
        self.l = list()
        self.l.append(0)
        for i in range(len(self.y)):
            self.q.append(self.p.sum(axis=1)[i])
            self.l.append(sum(self.q))

    def generate(self):
        x = np.random.uniform()
        k = r = 1
        for i in range(len(self.l) - 1):
            if self.l[i] < x <= self.l[i + 1]:
                for j in range(len(self.x)):
                    if sum(self.p[k - 1][0:j + 1]) + self.l[k - 1] < x:
                        r += 1
                    else:
                        break
                break
            else:
                k += 1
        return k - 1, r - 1
