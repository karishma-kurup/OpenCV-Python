# -*- coding: utf-8 -*-
"""


@author: user
"""

import cv2
import numpy as np


gray=cv2.imread('cup.jpg',0)
thresh=cv2.threshold(gray,127,255,cv2.THRESH_BINARY)[1]
otsus=cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]

kernel = np.ones((15, 15), np.uint8)
obj1 = cv2.dilate(otsus, kernel, iterations=1)
obj2 = cv2.erode(otsus, kernel, iterations=1)
obj3=cv2.morphologyEx(otsus,cv2.MORPH_OPEN,kernel)
obj4=cv2.morphologyEx(otsus,cv2.MORPH_CLOSE,kernel)

cv2.imshow('gray',gray)
cv2.imshow('thresh_binary',thresh)
cv2.imshow('Otsus',otsus)
cv2.imshow('dilate',obj1)
cv2.imshow('erode',obj2)
cv2.imshow('open',obj3)
cv2.imshow('close',obj4)
cv2.waitKey(0)
cv2.destroyAllWindows()    