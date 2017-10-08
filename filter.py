import numpy as np
import cv2

def dummy(val):
    pass

identity_kernel = np.array([[0,0,0],[0,1,0],[0,0,0]])
sharpen_kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
# use cv2 kernel
gaussian_kernel1 = cv2.getGaussianKernel(3,0)
gaussian_kernel2 = cv2.getGaussianKernel(5,0)
box_kernel = np.array([[1,1,1],[1,1,1],[1,1,1]], np.float32) / 9

kernels = [identity_kernel,sharpen_kernel,gaussian_kernel1,gaussian_kernel2,box_kernel]

windowName = 'app'
color_original = cv2.imread("test.jpg")
color_modified = color_original.copy()
gray_original = cv2.cvtColor(color_original,cv2.COLOR_BGR2GRAY)
gray_modified = gray_original.copy()

cv2.namedWindow(windowName)
cv2.createTrackbar('contrast',windowName,1,100,dummy)
cv2.createTrackbar('brightness',windowName,50,100,dummy)
cv2.createTrackbar('filter',windowName,0,len(kernels)-1,dummy)
cv2.createTrackbar('grayscale',windowName,0,1,dummy)

savecounter = 1

while True:
    grayscale = cv2.getTrackbarPos('grayscale',windowName)
    if grayscale == 0:
        cv2.imshow(windowName,color_modified)
    else:
        cv2.imshow(windowName,gray_modified)

    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'):
        break
    elif k == ord('s'):
        if grayscale == 0:
            cv2.imwrite('output%d.png' % savecounter,color_modified)
        else:
            cv2.imwrite('output%d.png' % savecounter,gray_modified)
        savecounter = savecounter + 1

    contrast = cv2.getTrackbarPos('contrast',windowName)
    brightness = cv2.getTrackbarPos('brightness',windowName)
    kernel = cv2.getTrackbarPos('filter',windowName)


    color_modified = cv2.filter2D(color_original,-1,kernels[kernel])
    gray_modified = cv2.filter2D(gray_original,-1,kernels[kernel])

    # Apply modifications to picture
    color_modified = cv2.addWeighted(color_modified,contrast,np.zeros(color_original.shape,dtype=color_original.dtype),0,brightness-50)
    gray_modified = cv2.addWeighted(gray_modified,contrast,np.zeros(gray_original.shape,dtype=gray_original.dtype),0,brightness-50)

cv2.destroyAllWindows()





#
# # if img is None:
# #     print "Image is not loaded"
# # else:
# #     print "Image is loaded"
#
# size = img.shape
# # make a gray scale picture
# gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# # apply contrast and brightness
# cb_img = cv2.addWeighted(img,4,np.zeros(img.shape,dtype=img.dtype),0,100)
# # apply kenel
# K= np.array([
#     [0,-1,0],
#     [-1,5,-1],
#     [0,-1,0]
# ])
# convolved = cv2.filter2D(img,-1,K)
#
# cv2.imshow('test',img)
# cv2.imshow('test1',gray)
# cv2.imshow('test2',cb_img)
# cv2.imshow('test3',convolved)
#
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
#
#
# # A = np.array([
# #     [1,1,2],
# #     [3,5,8]
# # ])
# # B = np.array([
# #     [2,3,5],
# #     [7,11,13]
# # ])
# # print A * 2
