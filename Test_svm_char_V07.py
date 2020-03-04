#!/usr/bin/env python
'''
SVM character detection.

Test_svm_char_V07.py can be tested on image as well as video.

It finds contours in the image or frame and selects samples from it.
Following preprocessing is applied to the image:
 - Moment-based image deskew (see deskew())
 - Digit images are split into 4 10x10 cells and 16-bin  histogram of
   oriented gradients is computed for each cell
 - Transform histograms to space with Hellinger metric (see [1] (RootSIFT))

SVM model used to predict value of samples, it returns a label which then
converted to ASCII value of the character and displayed as output (in yellow).
'''

import numpy as np
import cv2
import os
import sys
import video
from time import sleep
from train_svm_chars_v05 import *
global contour_max_height, contour_min_height, contour_max_width,contour_min_width
    


def onmouse(event, x, y, flags, param):
    '''
    Onmouse is used to capture the x y coordinates
    of the ROI as selected while dragging mouse
    '''
    global drag_start, sel ,UIlayer1
    if event == 1:                              # left mouse button down... started dragging 
        drag_start = x, y
        sel = 0,0,0,0
        #print "Drag Start",sel
    elif event == 4:                            # left mouse button up ... stop dragging
        drag_start = None
    elif drag_start:                            
        #print flags
        if flags and event==0:                    # ..continue dragging
            minpos = min(drag_start[0], x), min(drag_start[1], y)
            maxpos = max(drag_start[0], x), max(drag_start[1], y)
            sel = minpos[0], minpos[1], maxpos[0], maxpos[1]
            
        else:                                       # .. dragging completed
            drag_start = None


def recog_from_img(frame):
    '''
    recog_from_img detects all contours in the
    image selects possible characters and predicts their value
    '''
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)#get gray image
    bin = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 10)#adaptive thresholding
    bin = cv2.GaussianBlur(bin, (3,3),0)#blurring
    
    contours, heirs = cv2.findContours( bin.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)#find contours
                                                                                            #..i.e. probable charachters
    try: heirs = heirs[0]#prevent error of heirarchy
    except: heirs = []#...select only contours having no contours outside it
    result=[]
    for cnt, heir in zip(contours, heirs):#make separate list of contours and heirarchy then zip them together
        if heir[3]!=-1:
            continue
        
        x, y, w, h = cv2.boundingRect(cnt)#get bounding rectangle per contour
        #limiting the size to 
        #characters whose height is within 20-80% of frame height and max width of char is 1.8 time of its height
        if not (frame.shape[0]*0.2 <= h <= frame.shape[0]*.8  and w <= 1.8*h):
            continue 
        #provide padding for selected area
        xpad=int(w*.1)
        ypad=int(h*.05)
        x, w = int(x-xpad/2), w+xpad
        y, h = int (y-ypad/2), h+ypad
        #selecting the area containing possible character
        bin_roi = bin[y:,x:][:h,:w]
        gray_roi = gray[y:,x:][:h,:w]
        bin_roi1=bin_roi
        m = bin_roi != 0
        
        if not 0.3 < m.mean() < .8:
            continue
        #ROI is rotated and skewed prior to detection so as to enhance accuracy of recognition
        s = 1.5*float(h)/SZ
        m = cv2.moments(bin_roi)
        c1 = np.float32([m['m10'], m['m01']]) / m['m00']
        c0 = np.float32([SZ/2, SZ/2])
        t = c1 - s*c0
        A = np.zeros((2, 3), np.float32)
        A[:,:2] = np.eye(2)*s
        A[:,2] = t
       
        bin_norm = cv2.warpAffine(bin_roi1, A, (SZ, SZ), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
        bin_norm1 = deskew(bin_norm)
        sample = preprocess_hog([bin_norm1])
        label = model1.predict(sample)[0]#result in terms of label ie the row number of
                                         #char in the learning data set that is recognised in the test image
        
        if label>9:
            ch=chr(int(65+label-10))#getting ASCII value of character
        else:
            ch=str(int(label))#else print label for numbers 0-9
           
        res={"posXY":(x,y,w,h),"char":ch,"img":bin_norm1}
        result.append(res)

    result.sort(key=lambda x:x['posXY'][0]+x['posXY'][1])
    op=""
    for res in result:
        op=op+res['char']
    
    return(op,result)

    
    



def recog_from_video(frame):
    '''
    recog_from_video detects all contours in the video frame and selects possible characters
    It displays predicted result in the ROI frame
    '''
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)#get gray image
    bin = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 10)#adaptive thresholding
    bin = cv2.GaussianBlur(bin, (3,3),0)
    cv2.imshow('bin', bin)
    contours, heirs = cv2.findContours( bin.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)#find contours
    try: heirs = heirs[0]#prevent error of heirarchy
    except: heirs = []
    result=[]  
    for cnt, heir in zip(contours, heirs):#make separate list of contours and heirarchy ,zip them together
        if heir[3]!=-1:
            continue
        
        x, y, w, h = cv2.boundingRect(cnt)#get bounding rectangle per contour
        
        if not (frame.shape[0]*0.2 <= h <= frame.shape[0]*.8  and w <= 1.8*h):#limiting the size
            continue 
        
        #provide padding for selected area
        xpad=int(w*.1)
        ypad=int(h*.05)
        x, w = int(x-xpad/2), w+xpad
        y, h = int (y-ypad/2), h+ypad
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0))
        #selecting the area containing possible character
        bin_roi = bin[y:,x:][:h,:w]
        gray_roi = gray[y:,x:][:h,:w]
        bin_roi1 = bin_roi
        m = bin_roi != 0
                
        if not 0.3 < m.mean() < .8:
            continue
        #ROI is rotated and skewed prior to detection so as to enhance accuracy of recognition
        s = 1.5*float(h)/SZ
        m = cv2.moments(bin_roi)
        c1 = np.float32([m['m10'], m['m01']]) / m['m00']#calculate centroid
        c0 = np.float32([SZ/2, SZ/2])
        t = c1 - s*c0
        A = np.zeros((2, 3), np.float32)
        A[:,:2] = np.eye(2)*s
        A[:,2] = t
        bin_norm = cv2.warpAffine(bin_roi1, A, (SZ, SZ), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
        bin_norm1 = deskew(bin_norm)
        
        if x+w+SZ < frame.shape[1] and y+SZ < frame.shape[0]:
            frame[y:,x+(w/2):][:SZ, :SZ] = bin_norm[...,np.newaxis]
        sample = preprocess_hog([bin_norm1])
        #get result
        label = model1.predict(sample)[0]#result in terms of label ie the row number of
                                         #char in the learning data set that is recognised in the test image
        
        if label>9:
            ch=chr(int(65+label-10))
        else:
            ch=str(int(label))
            
        cv2.putText(frame, '%s'%ch, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 255), thickness = 2)
        res={"posXY":(x,y),"char":ch}
        result.append(res)
    result.sort(key=lambda x:x['posXY'][0]+x['posXY'][1])
    
    op=""
    for res in result:
        op=op+res['char']
    


def mainimg():
    '''Routine to recognise chars from images
        Test images are assumed to be in the current dir and follows
        a naming scheme of xx.jpg is followed, where xx is the image number
        12 such Test images are expected in the current dir
    '''
    global model1  #the SVM model isntance
    classifier_fn = 'charc_svm.dat'
    if not os.path.exists(classifier_fn):#if there is no machine learning file then create one
        print '"%s" not found, run train_svm_chars_v05.py first' % classifier_fn
        return
    model1 = SVM()
    model1.load(classifier_fn)
    disps=np.zeros((5,750), np.uint8)
    disps=cv2.cvtColor(disps,cv2.COLOR_GRAY2BGR)
    outputimg=np.zeros((50,750),np.uint8)
    outputimg=cv2.cvtColor(outputimg,cv2.COLOR_GRAY2BGR)
    cv2.putText(outputimg, "original Image        Possible regions     Recognised chars", (1, 20), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), thickness = 2)
    #cv2.imshow('OCR Header', header)
    #disps = np.concatenate(header, axis=1)
    for i in range(12):
        chrs=""
        img = cv2.imread(str(int(i+1))+'.jpg')  
        img=cv2.resize(img,(800,200))               #Input(Test) image resized
        bins=np.zeros((50,50), np.uint8)            #
        bins=cv2.cvtColor(bins,cv2.COLOR_GRAY2BGR)  #Placeholder for  binary ROI 
        frame=img.copy()                            #make a copy of original Test image
        possibleno,results=recog_from_img(frame)    #attempt to recognize the copy of test image
        reslt=np.zeros((50,250), np.uint8)          #placeholder for result(output) ie a sequence of chars after detection
        reslt=cv2.cvtColor(reslt,cv2.COLOR_GRAY2BGR)# placeholder for recognised chars
        for res in results:                         #for each sequence chars recognised in the output  
            bini=cv2.resize(res["img"],(50,50))     # resize for display...   
            bini=cv2.cvtColor(bini,cv2.COLOR_GRAY2BGR)# .. conver to color ( so that 
            bins=np.hstack((bins,bini))             #... and stack them horizontaly 
            chrs=chrs+res["char"]                   #Concatenate detected labels
        cv2.putText(reslt, '%s'%chrs, (10, 30), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 255, 255), thickness = 2)#...and print it
        disp=cv2.resize(bins,(250,50))              #resize binary frame containing all binary
                                                    #detected reigions of each test image....you never know how many char
                                                    #will be actualy dectected
        disp = np.concatenate((cv2.resize(img,(250,50)),disp), axis=1) # add  orginal test image to display frame
        disp = np.concatenate((disp,cv2.resize(reslt,(250,50))), axis=1)# add result 
        disps = np.concatenate((disps, disp), axis=0)                   #add detected bin ROIs
    outputimg=np.concatenate((outputimg,disps),axis=0)
    cv2.imshow('OCR Results', outputimg)                #display the complete frame of 12 tests
    
    while 1:   
        ch=cv2.waitKey(3000)
        if ch==27:                                  #break if esc key is pressed
            cv2.destroyAllWindows()
            break




           
def mainvid():
    '''
     mainvid detects possible characters from selected
    ROI and displays results in ROI frame
    '''
    global drag_start, sel ,UIlayer1,alpha,model1
    alpha='ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    drag_start=None 
    cap = video.create_capture(0)
    ret, frame = cap.read()
    sel = 0,0,0,0
    detectNow=False
    frozen=False
    #blank frames created
    blanklayer=np.zeros(frame.shape, np.uint8)
    ROI=np.zeros(frame.shape, np.uint8)
    #train_svm_chars_v05 .dat file loaded
    classifier_fn = 'charc_svm.dat'
    if not os.path.exists(classifier_fn):#if there is no machine learning file then create one
        print '"%s" not found, run train_svm_chars_v05.py first' % classifier_fn
        return
    #model selected
    model1 = SVM()
    model1.load(classifier_fn)
    cv2.namedWindow("frame",1)
    cv2.setMouseCallback("frame", onmouse)#if mouse is clicked call onmouse
    
    while True:
        if not frozen:
            ret, frame = cap.read()
        if (sel[0]!=sel[2]) and (sel[1]!=sel[3]):
            ROI=frame[sel[1]:sel[3],sel[0]:sel[2]].copy()#find ROI and make a copy
            
        ch = cv2.waitKey(1)
        if ch==32:
            frozen=not frozen
        if ch == 27:
            cap.release()
            cv2.destroyAllWindows() 
            break
            
        if not drag_start:
            recog_from_video(ROI)#detect characters in ROI 

        cv2.imshow('ROI', ROI)
        cv2.rectangle(frame, (sel[0], sel[1]), (sel[2], sel[3]), (0,200,250), 2)
        cv2.imshow('frame', frame)
         




if __name__ == '__main__':
    '''Either run image version or video version'''
    print __doc__
    mainimg()
    #mainvid()



