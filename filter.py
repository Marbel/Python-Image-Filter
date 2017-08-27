import numpy as np
import cv2

img = cv2.imread("test.jpg")
# if img is None:
#     print "Image is not loaded"
# else:
#     print "Image is loaded"

size = img.shape

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

cv2.imshow('test',img)
cv2.imshow('test1',gray)



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
