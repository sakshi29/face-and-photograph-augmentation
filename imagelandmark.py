# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 19:58:22 2016

@author: SAKSHI TRIPATHI
"""

import numpy as np
import cv2
import glob
import csv


ulist=[]
src_pts=[]
dst_pts=[]
f= open('foo1 (1).csv','rb')

img1=cv2.imread('arnie_30_30_200_200.jpg') 
img2=cv2.imread('arnie_30_30_200_2006.jpg')
h,w,c=img2.shape
reader = csv.reader(f, delimiter=',',quotechar='|')
for row in reader:
    ulist.append(row)
f.close()
 
for i in range (0,30,2):
            """     
            cv2.circle(img,(int(float(ulist[1][i])),int(float(ulist[1][i+1]))), 1, (0,0,255), 1)
            """
            src_pts.append([[np.float32(ulist[0][i]) ,np.float32(ulist[0][i+1])]])
            dst_pts.append([[np.float32(ulist[5][i]) ,np.float32(ulist[5][i+1])]])
src_pts=np.array(src_pts) 
dst_pts=np.array(dst_pts)             
print dst_pts           
M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0) 

##print M  
img2= cv2.warpPerspective(img2,M,(w,h))         
cv2.imwrite('maskit.jpg',img2)
cv2.waitKey(0)          