# coding: utf-8
import mxnet as mx
from mtcnn_detector import MtcnnDetector
import cv2
import os
import time
from scipy.misc import imread,imresize,imshow
from PIL import Image
import sys
import numpy as np

detector = MtcnnDetector(model_folder='./model', ctx=mx.cpu(0), num_worker = 1 , accurate_landmark = False)

#img = cv2.imread('test2.jpg')
img1 = imread(sys.argv[1])

# run detector
#img=cv2.resize(img1,(640,480))
img=img1
results = detector.detect_face(img)

if results is not None:

    total_boxes = results[0]
    points = results[1]

    # extract aligned face chips
    chips = detector.extract_image_chips(img, points, 144, 0.37)
    for i, chip in enumerate(chips):
        #cv2.imshow('chip_'+str(i), chip)
        #imshow(chip)
        #cv2.waitKey(0)
        cv2.imwrite('result/chip_'+str(i)+'.png', chip)

    draw = img.copy()
    for b in total_boxes:
        cv2.rectangle(draw, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255, 255, 255))

    for p in points:
        for i in range(5):
            cv2.circle(draw, (p[i], p[i + 5]), 1, (0, 0, 255), 2)

    imshow(draw)
    #cv2.waitKey(0)
else:
    print("no face detected")
# --------------
# test on camera
# --------------
'''
camera = cv2.VideoCapture(0)
cv2.namedWindow("detection result")
while True:
    grab, frame = camera.read()
    #img = cv2.resize(frame, (320,180))
    if frame:
        img = cv2.resize(frame, (320,180))
	#img=frame
    else:
        img=frame
	#frame=np.asarray(frame)
        #img = imresize(frame,(320,180))
    #img=frame
    #t1 = time.time()
    results = detector.detect_face(img)
    #print( 'time: ',time.time() - t1)

    if results is None:
        continue

    total_boxes = results[0]
    points = results[1]
    #img=np.mat(img)
    draw = img.copy()
    for b in total_boxes:
        cv2.rectangle(draw, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255, 255, 255))

    for p in points:
        for i in range(5):
            cv2.circle(draw, (p[i], p[i + 5]), 1, (255, 0, 0), 2)
    #cv2.namedWindow("detection result")
    cv2.imshow("detection result", draw)
    #draw=np.asarray(draw)
    #imshow(draw)
    #cv2.waitKey(30)
'''
