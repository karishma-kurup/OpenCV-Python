# -*- coding: utf-8 -*-
"""


@author: user
"""

import cv2

im=cv2.imread('landscape.jpg')
im_resize=cv2.resize(im,(512,512),interpolation=cv2.INTER_CUBIC)
im_resize2=cv2.resize(im,(32,32),interpolation=cv2.INTER_AREA)

img2=cv2.imread('cup.jpg')
im3=cv2.resize(im,(255,255),interpolation=cv2.INTER_AREA)
im4=cv2.resize(img2,(255,255),interpolation=cv2.INTER_AREA)

dst1=cv2.addWeighted(im3,0.6,im4,0.4,0)

dst2=cv2.bitwise_and(im3,im4)
dst3=cv2.bitwise_or(im3,im4)

dst4=cv2.bitwise_not(im)

cv2.imshow("im",im)
cv2.imshow('zoom',im_resize)
cv2.imshow('shrink',im_resize2)
cv2.imshow('weighted_addition',dst1)
cv2.imshow('and',dst2)
cv2.imshow('or',dst3)
cv2.imshow('not',dst4)
cv2.waitKey(0)
cv2.destroyAllWindows()