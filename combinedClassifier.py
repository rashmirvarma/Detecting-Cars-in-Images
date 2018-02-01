'''
Created on Nov 8, 2017

@author: Chinmay Mishra
'''

import cv2,os

cascadeClassifier = cv2.CascadeClassifier("cars.xml")

def classifyImage(imgPath,outputPath):
    '''
    Function that applies the Haar classifier to the image given by the user and returns the result and writes the images to a particular folder.
    '''
    img = cv2.imread(imgPath)
    result = "No car present"
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    cars = cascadeClassifier.detectMultiScale(img_gray,1.3,5)
    if len(cars)>0:
        print "car"
        result = "Car present"
        template = cv2.imread("carsFolder/testTrain2.jpg",0)
        w,h = template.shape[::-1]
        method = "cv2.TM_CCOEFF"
        method = eval(method)
        try:
            res = cv2.matchTemplate(img_gray,template,method)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        except Exception as e:
            min_val= max_val= min_loc= max_loc =0
        if min_val != 0:
            top_left = max_loc
            bottom_right = (top_left[0]+w,top_left[1]+h)
            cv2.rectangle(img,top_left,bottom_right,(0,255,0),10)
        img = cv2.resize(img, (960, 540))
    cv2.imwrite(outputPath,img)
    return result

    
