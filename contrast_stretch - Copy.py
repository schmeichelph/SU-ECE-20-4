# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 21:37:25 2019

@author: caballe4
"""

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

img = Image.open("C:/Users/caballe4/Desktop/01__Station05__Camera1__2012-6-14__5-38-10(2).jpg")

multi = img.split()
red_band = multi[0].point(red)
green_band = multi[1].point(green)
blue_band = multi[2].point(blue)

normal_img = Image.merge("RGB",(red_band, green_band, blue_band))

normal_img.show()

normal_img.save("C:/Users/caballe4/Desktop//01__Station05__Camera1__2012-7-16__15-14-30(2b).jpg", "JPEG")