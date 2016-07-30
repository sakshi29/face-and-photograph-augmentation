import numpy as np
import cv2

img1=cv2.imread('arnie_30_30_200_200.jpg') 
img2=cv2.imread('arnie_30_30_200_2006.jpg')
img3=cv2.imread('catgrabcut_output.jpg')
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
gray3=cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
x,y,c=img1.shape


for i in range (0,y-1):
    for j in range (0,x-1):
         if(gray3[j,i]==255):
            img1[j,i]=img2[j,i]
         else:
            continue

         

cv2.imshow('img',img1)    
cv2.waitKey(0)
cv2.destroyAllWindows()
