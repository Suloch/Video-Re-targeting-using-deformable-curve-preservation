
import cv2
import numpy as np
import math
import operator

#defininf dictionary to convert oriented segments into co-ordinate points
segment = {
            1 : [-2, -2],
            2 : [-1, -2],
            3 : [0, -2],
            4 : [1, -2],
            5 : [2, -2],
            6 : [2, -1],
            7 : [2, 0],
            8 : [2, 1],
            9 : [2, 2],
            10 : [1, 2],
            11 : [0, 2],
            12 : [-1, 2],
            13 : [-2, 2],
            14 : [-2, 1],
            15 : [-2, 0],
            16 : [-2, -1]
}


#function to calculate weight of a curve
#c -> curve I -> gradient magnitude of Image
#l -> lambda constant for smoothness term
#g -> gamma constant for length term
def weight(c, I):
    l = 5
    g = 0.5
    return F(c.points,I) + l*T(c.segments) + g*L(c.points)

#Intensity term of the weight of the curve
def F(C, I):
    f = 0
    for x in C:
        f = f + I[x[0]][x[1]]
    f = f / len(C)
    return f
#Smoothness term for the weight of the curve
def T(S):
    if len(S) < 3:
        return 0

    Th = 2
    value = 0
    n = len(S) - 1
    for i in range(n):
        value = value + t(S[i],S[i+1])

    value = value / n
    return value

#function t for calculating smoothness
def t(s, s1):
    Th = 2
    if s1 == 0:
        return 0
    value = math.hypot(segment[s][0]-segment[s1][0], segment[s][1]-segment[s1][1])

    if value >= Th:
        return math.exp(-1*value)
    else:
        return -float('inf')

#length term for the weight of the curve
def L(C):
    return math.log(len(C))

#function to calculate deformation between to curves
#cB -> representation of curve in Bookstein co-ordinates
def deformation(cB1, cB2):
    if len(cB1) == 0 or len(cB2) == 0:
        return float('inf')
    sumc1 = 0
    for xi in cB1:
        min_dist = float('inf')
        for xj in cB1:
            dist = math.hypot(xi[0]-xj[0], xi[1]-xj[1])
            if dist < min_dist:
                min_dist = dist
        sumc1 = sumc1 + min_dist
    sumc1 = sumc1 / (2*len(cB1))
    sumc2 = 0
    for xi in cB2:
        min_dist = float('inf')
        for xj in cB2:
            dist = math.hypot(xi[0]-xj[0], xi[1]-xj[1])
            if dist < min_dist:
                min_dist = dist
        sumc2 = sumc2 + min_dist
    sumc2 = sumc2 / (2*len(cB2))
    return sumc2 + sumc1

#CP -> curve in cartesian points form
def centroid(CP):
    c = np.array(CP)
    sumx = np.sum(c[:,0])
    sumy = np.sum(c[:,1])
    sumx = sumx / c.shape[0]
    sumy = sumy / c.shape[0]

    return [sumx, sumy]

#calculate centroid_cost
def centroid_cost(cen1, cen2):
    Tc = 20
    dist = math.hypot(cen1[0]-cen2[0], cen1[1]-cen2[1])

    if dist < Tc:
        return dist/Tc
    return float('inf')

#calculate scale cost
def scale(C1, C2, cen1, cen2):
    Tr = 0.8

    c1 = np.array(C1)
    rc1 = np.linalg.norm(c1-cen1)
    rc1 = rc1/len(C1)

    c2 = np.array(C2)
    rc2 = np.linalg.norm(c2-cen2)
    rc2 = rc2/len(C2)

    minrc = min(rc1, rc2)
    maxrc = max(rc1, rc2)
    t = minrc/maxrc
    if t > Tr:
        return t
    return float('inf')

#calculate orientation cost
def orientation(C1, C2):
    To = 0.2
    h1 = calculateHistogram(C1)
    h2 = calculateHistogram(C2)
    value = np.linalg.norm(h1-h2)

    if value <= To:
        return value
    return float('inf')

#calculate histogram for orientation cost
#s -> curve in segments form
def calculateHistogram(S):
    o = []
    for si in S:
        o.append((math.atan2(segment[si][1], segment[si][0])+math.pi))

    bins = np.linspace(0, 2*math.pi, 9)
    h = np.zeros(9)

    for oi in o:
        index = np.argmax(bins > oi)
        value = (4/math.pi)*(bins[index]-oi)
        h[index] = h[index]+value
        h[index-1] = h[index-1]+1-value

    h_normalised = h/np.sum(h)

    return h_normalised

#class to represent a curve
class Curve:
    def __init__(self):
        self.points=[]
        self.segments=[]
        self.bookstein=[]
        self.weight=0
#adding new points to curve
    def addPoint(self, point):
        for p in self.points:
            if p[0]==point[0] and p[1]==point[1]:
                return False
        self.points.append([int(point[0]),int(point[1])])
        s = []
        s.append(point[0]-self.points[-1][0])
        s.append(point[1]-self.points[-1][1])
        for i,v in segment.items():
            if v[0]==s[0] and v[1]==s[1]:
                self.segments.append(i)
#adding new segments to the curves
#returns false if the segment forms a loop with the curve
    def addSegment(self, s):
        point=[]
        point.append(self.points[-1][0]+segment[s][0])
        point.append(self.points[-1][1]+segment[s][1])

        for point1 in self.points:
            if point[0]==point1[0] and point[1]==point1[1]:
                return False
        self.points.append(point)
        self.segments.append(s)
        return True

#assign weight to the curve
    def assignWeight(self, I):
        self.weight=weight(self,I)

#calculate the bookstein co-ordinates
    def transformBookstein(self):

        self.bookstein = []
        x1 = self.points[0][0]
        y1 = self.points[0][1]
        xn = self.points[-1][0]
        yn = self.points[-1][0]

        if x1 == xn and y1 == yn:
            return
        for i,point in enumerate(self.points):
            xi = self.points[i][0]
            yi = self.points[i][1]

            xb = ((xn-x1)*(xi-x1) + (yn-y1)*(yi-y1)) / (((xn-x1)**2)+((yn-y1)**2))
            xb = xb - 0.5

            yb = ((xn-x1)*(yi-y1) + (yn-y1)*(xi-x1)) / (((xn-x1)**2)+((yn-y1)**2))

            self.bookstein.append([xb,yb])

#calculate the matching cost with another curve
    def matchCost(self, c):
        cen1 = centroid(self.points)
        cen2 = centroid(c.points)
        self.transformBookstein()
        c.transformBookstein()
        cost = deformation(self.bookstein, c.bookstein) + centroid_cost(cen1,cen2)
        cost = cost + scale(self.points,c.points,cen1,cen2) + orientation(self.segments,c.segments)
        return cost

#deform the curve using a seam
    def deform(self, seam):
        for point in self.points:
            if point[1] <= seam[point[0]]:
                point[1] = point[1] - 1

#function to extract curve
def extractCurve(gray_I, grad_mag):
    corners = cv2.goodFeaturesToTrack(gray_I, 10, 0.01, 5)
    if type(corners) == 'NoneType':
        return []

    curves = []

    for corner in corners:
        if corner[0][0] < grad_mag.shape[0] and corner[0][1] < grad_mag.shape[1]:
            c = Curve()
            c.addPoint(corner[0])

            curves.append(c)
            x = corner[0]
            s_checked = []
            while True:
                s = findmaxs(x, c, grad_mag, s_checked)
                if s == -1 or s == 0:
                    break
                if c.addSegment(s) == False:
                    if len(s_checked) == 15:
                        break
                    else:
                        s_checked.append(s)
                    continue
                x = c.points[-1]
            c.assignWeight(grad_mag)

    curves.sort(key=operator.attrgetter('weight'),reverse=True)
    salientcurves = []
    for c in curves:
        for c1 in curves:
            if c!= c1:
                for point in c.points:
                    flag = False
                    for point1 in c1.points:
                        if point[0]==point1[0] and point[1]==point1[1]:
                            curves.remove(c1)
                            flag = True
                            break
                    if flag == True:
                        break
        salientcurves.append(c)
    return salientcurves

#function to find s such that weight is maximum
def findmaxs(corner, c, I, s_checked):
    max_value = -float('inf')
    maxs = -1

    for s in segment:
        flag = False
        if s not in s_checked:
            l = 5
            x = int(corner[0]) + segment[s][0]
            y = int(corner[1]) + segment[s][1]
            value = -float('inf')
            if x >= 0 and x < I.shape[0] and y >= 0 and y < I.shape[1]:
                if len(c.points) < 2:
                    s1 = 0
                else:
                    s1 = c.segments[-1]
                value = float(I[x][y]) + l*t(s, s1)
            if value > max_value:
                max_value = value
                maxs = s
    return maxs
