# -*- coding: utf-8 -*-
"""
@author: user
"""

import cv2
q
cap=cv2.VideoCapture(0)
while True:
    ret,frame=cap.read()
    cv2.imshow('frame',frame)
    print(cap.get(3))
    print(cap.get(4))
    if cv2.waitKey(1)&0xFF==ord("q"):
        break
cap.release()
cv2.destroyAllWindows()