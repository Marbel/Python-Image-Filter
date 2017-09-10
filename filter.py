import numpy as np
import cv2

img = cv2.imread("test.jpg")
# if img is None:
#     print "Image is not loaded"
# else:
#     print "Image is loaded"

size = img.shape
# make a gray scale picture
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# apply contrast and brightness
cb_img = cv2.addWeighted(img,4,np.zeros(img.shape,dtype=img.dtype),0,100)
# apply kenel
K= np.array([
    [0,-1,0],
    [-1,5,-1],
    [0,-1,0]
])
convolved = cv2.filter2D(img,-1,K)

cv2.imshow('test',img)
cv2.imshow('test1',gray)
cv2.imshow('test2',cb_img)
cv2.imshow('test3',convolved)


cv2.waitKey(0)
cv2.destroyAllWindows()



# A = np.array([
#     [1,1,2],
#     [3,5,8]
# ])
# B = np.array([
#     [2,3,5],
#     [7,11,13]
# ])
# print A * 2
