# -*- coding: utf-8 -*-
"""


@author: user
"""

import cv2
img=cv2.imread('cup.jpg')
img=cv2.line(img,(0,0),(255,255),(255,0,0),3)
img=cv2.rectangle(img,(38,56),(150,120),(0,255,0),3)
img=cv2.circle(img,(63,63),25,(0,255,0),-1)
font=cv2.FONT_HERSHEY_SIMPLEX
img=cv2.putText(img,'draw',(255,255),font,0.5,(0,0,255),10,cv2.LINE_AA)
cv2.imshow('out',img)
cv2.waitKey(0)
cv2.destroyAllWindows()