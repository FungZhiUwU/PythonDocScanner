#Import
import cv2
import numpy as np
import pytesseract as pt

#Define the path of tesseract.exe
# -----------------Remember to change the path of the tesseract.exe--------------------
pt.pytesseract.tesseract_cmd=r"D:\\Programmer_Life\\Opencv\\tesseract.exe"

#Define the size of the image
heightImg = 640
widthImg  = 480

#Passing the slider and button value
def something(x):
    pass

#Reorder the points for warp
def reorder(imgPoint):
 
    imgPoint = imgPoint.reshape((4, 2))
    imgPointReorder = np.zeros((4, 1, 2), dtype=np.int32)
    add = imgPoint.sum(1)
 
    imgPointReorder[0] = imgPoint[np.argmin(add)]
    imgPointReorder[3] =imgPoint[np.argmax(add)]
    diff = np.diff(imgPoint, axis=1)
    imgPointReorder[1] =imgPoint[np.argmin(diff)]
    imgPointReorder[2] = imgPoint[np.argmax(diff)]
 
    return imgPointReorder

# Setup
#-----------------Remember to change the path of the image and the File name and type-----------------
imgLocation = "test_img.jpg" #Input the image name
img = cv2.imread('D:\\Programmer_Life\\Opencv\\Assignement\\'+imgLocation)
img = cv2.resize(img,(widthImg,heightImg),cv2.INTER_AREA)
BW_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
Blur_img = cv2.GaussianBlur(BW_img, (5, 5), 1)

#Create Window
cv2.namedWindow('Threshold')
cv2.namedWindow('Contour')
cv2.namedWindow('CornerFrame')
cv2.namedWindow('Sharpen')
cv2.namedWindow('OCR')

#Slider
cv2.createTrackbar('Threshold','Threshold',0,255,something) #Place the threshold slider at "Threshold" window

#Switch
switch = "0 : OFF \n 1 : ON" # create switch for ON/OFF functionality
cv2.createTrackbar(switch, 'Threshold',0,1,something)  #Place the switch at "Threshold" window

while(1):

    s = cv2.getTrackbarPos (switch, 'Threshold') #Get the switch value
    t = cv2.getTrackbarPos ('Threshold', 'Threshold') #Get the threshold value

    if s == 0:
        # Set the window to display the original image when the switch is off
        img=img
        thresh = img      
        ContourFrame=img
        CornerFrame = img
        imgWarpColored = img
        Sharpen = img
        Final_img = img

        # ====== No Threshold Option ======
        imgWarp_txt = img
        
        # ====== Threshold Optiion ======
        # ocr_thresh = img

    if s == 1:
        r, thresh = cv2.threshold(Blur_img,t,255,cv2.THRESH_BINARY) #Threshold the image
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #Find the contours
        ContourFrame = img.copy() 
        ContourFrame = cv2.drawContours(ContourFrame, contours, -1, (255, 0, 255), 3) #Draw the contours
        CornerFrame = img.copy()

        maxArea = 0
        biggest = [] # PREPARE A NEW ARRAY TO STORE ALL THE BIGGEST CONTOUR
        for i in contours :
            area = cv2.contourArea(i)
            if area > 100:
                peri = cv2.arcLength(i, True)
                edges = cv2.approxPolyDP(i, 0.02*peri, True)
                if area > maxArea and len(edges) == 4 : #Check if the contour is rectangle
                    biggest = edges
                    maxArea = area
                               
        if len(biggest) != 0 :
            # REORDER FOR WARPING
            biggest=reorder(biggest)
            CornerFrame = cv2.drawContours(CornerFrame, biggest, -1, (255, 0, 255), 25) # DRAW THE BIGGEST CONTOUR
            
            # WARP THE IMAGE
            pts1 = np.array(biggest, np.float32) # PREPARE POINTS FOR WARP
            pts2 = np.float32([[0, 0],[widthImg, 0], [0, heightImg],[widthImg, heightImg]]) # PREPARE POINTS FOR WARP
            matrix = cv2.getPerspectiveTransform(pts1, pts2) 
            imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg)) # WARP THE IMAGE
            
            Final_img = imgWarpColored.copy()
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])  # Kernel for sharpening the image
            Sharpen = cv2.filter2D(Final_img, -1, kernel) #APPLY Sharpening

            # ---------- Pytessearact ----------
            imgWarp_txt = cv2.warpPerspective(img, matrix, (widthImg, heightImg))

            # ====== Threshold Option ======
            # kernel_txt = cv2.getStructuringElement(cv2.MORPH_RECT, (1,2))
            # BW_txt = cv2.cvtColor(imgWarp_txt,cv2.COLOR_BGR2GRAY)
            # morphology_img = cv2.morphologyEx(BW_txt, cv2.MORPH_OPEN, kernel_txt,iterations=2)          
            # ocr_thresh = cv2.adaptiveThreshold(morphology_img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)

            txt = pt.image_to_string(imgWarp_txt)
            print("=========== Start =========== \n "+ txt + "=========== End ===========")
   
    cv2.imshow("Original", img)
    cv2.imshow('Threshold', thresh)
    cv2.imshow('Contour',ContourFrame)
    cv2.imshow('CornerFrame',CornerFrame)
    cv2.imshow('Final', imgWarpColored)
    cv2.imshow('Sharpen',Sharpen)

    # ====== Either thresh or no thresh ====== 
    # ====== No Threshold Option ======
    cv2.imshow('OCR',imgWarp_txt)
    
    # ====== Threshold Optiion ======
    # cv2.imshow('OCR',ocr_thresh)
        
    k = cv2.waitKey(1) & 0xFF
    if k==27:
        break
cv2.destroyAllWindows()    