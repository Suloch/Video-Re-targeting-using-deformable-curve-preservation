
import math
import numpy as np
import curve


def Es(seam, Ms):
    sum = 0
    H = Ms.shape[0]
    for i in range(H):
        sum = sum + Ms[i][seam[i]]

    sum = sum / H

    return sum

def Ed(seam, C):
    value = 0
    for c in C:
        new_c = curve.Curve()
        for point in c.points:
            new_point = point
            if point[1] < seam[point[0]]:
                new_point[1] = new_point[1] - 1

            new_c.addPoint(new_point)

        c.transformBookstein()
        new_c.transformBookstein()
        value = value + curve.deformation(c.bookstein, new_c.bookstein)

    return value

def findMatch(C1, C2):
        match = []
        Td = 4

        for i,cur1 in enumerate(C1):
            min_cost = float('inf')
            for j,cur2 in enumerate(C2):
                d = cur1.matchCost(cur2)

                if d < min_cost:
                    min_cost = d

            if min_cost < Td:
                match.append([i,j])
        return match

def calculateR(C1, C2, match):
    M = []
    #print(match)
    for m in match:
        mvec = []
        for point1 in C1[m[0]].points:
            min_k = 0
            min_d = float('inf')
            for k,point2 in enumerate(C2[m[1]].points):
                d = math.hypot(point1[0]-point2[0], point1[1]-point2[1])
                if d < min_d:
                    min_d = d
                    min_k = k
            mvec.append(min_k)
        M.append(mvec)

    R = []
    for i,m in enumerate(match):
        r = []
        c1 = C1[m[0]]
        c2 = C2[m[1]]
        for j,point in enumerate(c1.points):

            if c2.points[M[i][j]][1] < point[1]:
                r.append(1)
            elif c2.points[M[i][j]][1] == point[1]:
                r.append(0)
            else:
                r.append(-1)
        R.append(r)
    return R

def Et(RW, rw1):
    N = len(RW)
    if N == 0:
        return 0

    sum_ = 0
    for i in range(N):
        r = np.abs(np.array(RW[i]) - np.array(rw1[i]))
        sum_ = sum_ + np.sum(r)

    return sum_/N

class Seam:
    def __init__(self, Ms):
        self.Ms = Ms
        self.Mt, self.Mc = self.calculateEnergyMap(Ms)


    def extractSeams(self):
        P = []
        X = len(self.Mc)
        Y = len(self.Mc[0])
        for j in range(Y):
            p = np.zeros(X, dtype=np.int32)
            p[X-1] = j
            for i in range(X-2, -1, -1):
                p[i] = p[i+1] + self.Mt[i+1][p[i+1]]
            P.append(p)
        Ts = 5

        count = 0
        for i,pi in enumerate(P):
            flag = True
            for j,pj in enumerate(P):
                if i != j:
                    value = np.sum(abs(pi-pj))
                    value = value / X
                    if value <= Ts:
                        P.pop(j)


        return P

    def calculateEnergyMap(self, Ms):
        Mt = np.zeros(Ms.shape)
        Mc = np.zeros(Ms.shape)
        X = len(Ms)
        Y = len(Ms[0])

        Mc[0] = Ms[0]
        for i in range(1,X):
            for j in range(Y):
                min_k = 0
                min_v = float('inf')
                for k in[-1 ,0,1]:
                    y = j + k
                    x = i - 1
                    if y < 0 or y >= Y:
                        continue

                    if Mc[x][y] < min_v:
                        min_v = Mc[x][y]
                        min_k = k
                Mc[i][j] = Ms[i][j] + min_v
                Mt[i][j] = min_k


        return Mt, Mc
