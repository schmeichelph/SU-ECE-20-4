# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 10:26:32 2019

@author: caballe4
"""

#from _future_ import print_function
import cv2 as cv2
from PIL import Image

def red(intensity):
    iI = intensity
    i_min = 86
    i_max = 230
    
    o_min = 0
    o_max = 255
    
    io = ((iI-i_min)/(i_max-i_min))*255
    #io = (iI-i_min)*(((o_max-o_min)/(i_max - i_min)) + o_min)
    return io

def green(intensity):
    iI = intensity
    i_min = 90
    i_max = 225
    
    o_min = 0
    o_max = 255
    
    io = ((iI-i_min)/(i_max-i_min))*255
    #io = (iI-i_min)*(((o_max-o_min)/(i_max - i_min)) + o_min)
    return io

def blue(intensity):
    iI = intensity
    i_min = 100
    i_max = 210
    
    o_min = 0
    o_max = 255
    
    io = ((iI-i_min)/(i_max-i_min))*255
    #io = (iI-i_min)*(((o_max-o_min)/(i_max - i_min)) + o_min)
    return io

img = Image.open("C:/Users/caballe4/Desktop/01__Station4__Camera2__2012-07-22__00-53-20(1).jpg")

multi = img.split()
red_band = multi[0].point(red)
green_band = multi[1].point(green)
blue_band = multi[2].point(blue)

normal_img = Image.merge("RGB",(red_band, green_band, blue_band))

normal_img.show()

normal_img.save("C:/Users/caballe4/Desktop/01__Station05__Camera1__2012-7-16__15-14-30(10c).jpg", "JPEG")

image1 = cv2.imread('C:/Users/caballe4/Desktop/01__Station05__Camera1__2012-7-16__15-14-30(10c).jpg')
cv2.imshow("original",image1)
image1 = image1.astype('uint8')
image1 = cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY)

hist_image = cv2.equalizeHist(image1)

cv2.imshow("Original Image",image1)
cv2.imshow("Equal Image", hist_image)
cv2.imwrite('C:/Users/caballe4/Desktop/01__Station05__Camera1__2012-7-16__15-14-30(104).jpg',hist_image)
cv2.waitKey()