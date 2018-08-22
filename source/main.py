
import cv2
import numpy as np
import curve
import seams
import math
import sys
import time
import os

directory = ''

def videoToFrames(path):
    video = cv2.VideoCapture(path)
    frames = []
    i = 0
    while True:
        ret, frame = video.read()
        i = i + 1
        if ret == False or i == 3:
            break
        frames.append(frame)

    video.release()
    return frames

def framesToVideo(frames, path):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    x = frames[0].shape[0]
    y = frames[1].shape[1]
    out = cv2.VideoWriter(path, fourcc, 30.0, (y,x))
    i = 0
    for frame in frames:
        i = i + 1
        cv2.imwrite(directory+"matlab/fra%d.tiff" %i, frame)
        out.write(frame)

    out.release()

def gray(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

def gradientMagnitude(frame):
    frame_gray = gray(frame)

    sobelx = cv2.Sobel(frame_gray, cv2.CV_64F, 1, 0, ksize=3)
    ab_sobelx = cv2.convertScaleAbs(sobelx)
    sobely = cv2.Sobel(frame_gray, cv2.CV_64F, 0, 1, ksize=3)
    ab_sobely = cv2.convertScaleAbs(sobely)
    mag = cv2.addWeighted(ab_sobelx, 0.5, ab_sobely, 0.5, 0)

    return mag

def spectralResidual(frame):
    frame_gray = gray(frame)
    dft = cv2.dft(np.float32(frame_gray), flags=cv2.DFT_COMPLEX_OUTPUT)
    A, P = cv2.cartToPolar(dft[:,:,0],dft[:,:,1])
    L = np.log(A)
    kernel = np.ones((3,3),np.float32)/9
    new_L = cv2.filter2D(L, -1, kernel)
    R = new_L - L
    new_dft = np.empty(dft.shape)
    new_dft[:,:,0], new_dft[:,:,1] = cv2.polarToCart(R,P)
    Mr = cv2.idft(new_dft)
    MrI = cv2.magnitude(Mr[:,:,0],Mr[:,:,1])

    return MrI

def removeSeam(frame, seam):
    newframe = np.copy(frame)
    for i in range(frame.shape[0]):
        for j in range(seam[i],frame.shape[1]-1):
            newframe[i][j] = frame[i][j+1]

    newframe = np.delete(newframe, -1, 1)
    return newframe

name = sys.argv[1].partition('.')[0]
directory = "../final2Output/"+name+"/frames/"
directory2 = "../final2Output/"+name+"/seams/"
n_seams = 240 - int(sys.argv[2])
if not os.path.exists(directory):
    os.makedirs(directory)
if not os.path.exists(directory2):
    os.makedirs(directory2)

f = open("../final2Output/"+name+"/profile.txt", "w+")
f.write("%-25s %-10s\n" %("Process","Time"))
input_video = "../input/ExampleVideos/"+sys.argv[1]
output_video = "../final2Output/"+name+".avi"

frames = videoToFrames(input_video)
N = len(frames)

t1 = time.process_time()
Curves = []
print('Detecting curves in frames...')


for i, frame in enumerate(frames):
    print('Frame', i, end=' ')
    Curves.append(curve.extractCurve(gray(frame),gradientMagnitude(frame)))

t2 = time.process_time()
f.write("%-25s %-10.2f\n" %("Curve Detection",t2-t1))
print()
"""
for i,Curve in enumerate(Curves):
    frame_copy = np.copy(frames[i])
    for c in Curve:
        for point in c.points:
            cv2.circle(frame_copy, (point[1],point[0]), 1, 255, -1)
    cv2.imwrite("../output/rat/frames/keypoints/frame%d.jpg" %i, frame_copy )
"""
t1 = time.process_time()
RW = []
Match = []
print('Matching Frames...')
for i in range(1, N):
    match = seams.findMatch(Curves[i-1],Curves[i])
    RW.append(seams.calculateR(Curves[i-1], Curves[i], match))
    Match.append(match)
    print('Frame', i, end=' ')
t2 = time.process_time()
f.write("%-25s %-10.2f\n" %("Curve Matching",t2-t1))
print()

t_est = time.process_time()
for w in range(240, n_seams, -1):

    for i in range(1,N):
        t1 = time.process_time()
        Mg = gradientMagnitude(frames[i])
        Mr = spectralResidual(frames[i])
        Ms = 0.4*Mr + 0.6*Mg
        P = seams.Seam(Ms).extractSeams()

        min_p = -1
        min_E = float('inf')
        for j,p in enumerate(P):
            Es = seams.Es(p, Ms)
            Ed = seams.Ed(p, Curves[i])

            n_Curves = Curves[i]
            n_Curves1 = Curves[i-1]
            for x in n_Curves:
                x.deform(p)
            for x in n_Curves1:
                x.deform(p)
            rw1 = seams.calculateR(n_Curves1, n_Curves, Match[i-1])
            Et = seams.Et(RW[i-1], rw1)

            E = Es + 15*Ed + 5*Et

            if E < min_E:
                min_E = E
                min_p = j

            t2 = time.process_time()
        f.write("%f, " %(t2-t1))

        frame_copy = np.copy(frames[i])
        for c in Curves[i]:
            for point in c.points:
                cv2.circle(frame_copy, (point[1],point[0]), 1, (0,0,255), -1)
        for k,point in enumerate(P[min_p]):
            cv2.circle(frame_copy, (point,k), 1, (0,255,0), -1)
        cv2.imwrite(directory+"frame%d,%d.jpg" %(i, 241-w), frame_copy )


        l_est = time.process_time() - t_est
        print('optimum seam', min_p, "Frame Number", i, "Estimated time left:", int(-l_est*(1-1/(((240-w)*N+i)/(N*int(sys.argv[2]))))))
        frames[i] = removeSeam(frames[i], P[min_p])
        cv2.imwrite(directory+"frame_%d,%d.jpg" %(i, 241-w), frames[i] )

        if w == 240:
            f_seam = open(directory2+"Frame%d.txt" %(i),"w")
        else:
            f_seam = open(directory2+"Frame%d.txt" %(i), "a")
        f_seam.write(np.array2string(P[min_p], separator=','))
        f_seam.write("\n\n")
        f_seam.close()

        for c in Curves[i]:
            c.deform(P[min_p])
    f.write("\n")
framesToVideo(frames, output_video)
f.close()
