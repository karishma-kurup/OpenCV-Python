# -*- coding: utf-8 -*-
"""


@author: user
"""
import cv2
import numpy as np

im=cv2.imread('cup.jpg')
lap=cv2.Laplacian(im,cv2.CV_64F,ksize=3)
lap=np.uint8(np.absolute(lap))
sobelx=cv2.Sobel(im,cv2.CV_64F,0,1)
sobelx=np.uint8(np.absolute(sobelx))
sobely=cv2.Sobel(im,cv2.CV_64F,1,0)
sobely=np.uint8(np.absolute(sobely))
sobel_combine=cv2.bitwise_or(sobelx,sobely)
canny=cv2.Canny(im,125,255)

cv2.imshow('im',im)
cv2.imshow('lap',lap)
cv2.imshow('sobel_x',sobelx)
cv2.imshow('sobel_y',sobely)
cv2.imshow('canny',canny)
cv2.waitKey(0)
cv2.destroyAllWindows()