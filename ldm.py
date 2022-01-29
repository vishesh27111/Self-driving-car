import cv2
import utlis
import numpy as np

curveList=[]
avgVal=10

def getLaneCurve(img,display=2):
    ImgCopy = img.copy()
    ImgResult = img.copy()
    #step1
    ImgThres = utlis.thresholding(img)
    #step2
    hT,wT,c=  img.shape
    points=utlis.valTrackbars()
    ImgWarp=utlis.warpImg(ImgThres, points, wT, hT)	
    ImgWarpPoints = utlis.drawPoints(ImgCopy, points)
    #step3
    midPoint,ImgHist= utlis.getHistogram(ImgWarp,display=True,minPer=0.5,region=4)
    curveAveragePoint,ImgHist= utlis.getHistogram(ImgWarp,display=True,minPer=0.9)
    curveRaw = curveAveragePoint - midPoint
    #step4
    curveList.append(curveRaw)
    if len(curveList) >= avgVal:
        curveList.pop(0)
    curve = int(sum(curveList)/len(curveList))
    #step5
    if display != 0:
       imgInvWarp = utlis.warpImg(ImgWarp, points, wT, hT,inv = True)
       imgInvWarp = cv2.cvtColor(imgInvWarp,cv2.COLOR_GRAY2BGR)
       imgInvWarp[0:hT//3,0:wT] = 0,0,0
       ImgLaneColor = np.zeros_like(img)
       ImgLaneColor[:] = 0, 255, 0
       ImgLaneColor = cv2.bitwise_and(imgInvWarp, ImgLaneColor)
       ImgResult = cv2.addWeighted(ImgResult,1,ImgLaneColor,1,0)
       midY = 450
       cv2.putText(ImgResult,str(curve),(wT//2-80,85),cv2.FONT_HERSHEY_COMPLEX,2,(255,0,255),3)
       cv2.line(ImgResult,(wT//2,midY),(wT//2+(curve*3),midY),(255,0,255),5)
       cv2.line(ImgResult, ((wT // 2 + (curve * 3)), midY-25), (wT // 2 + (curve * 3), midY+25), (0, 255, 0), 5)
       for x in range(-30, 30):
           w = wT // 20
           cv2.line(ImgResult, (w * x + int(curve//50 ), midY-10),
                    (w * x + int(curve//50 ), midY+10), (0, 0, 255), 2)
       #fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
       #cv2.putText(ImgResult, 'FPS '+str(int(fps)), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (230,50,50), 3);
    if display == 2:
       imgStacked = utlis.stackImages(0.7,([img,ImgWarpPoints,ImgWarp],
                                         [ImgHist,ImgLaneColor,ImgResult]))
       cv2.imshow('ImageStack',imgStacked)
    elif display == 1:
       cv2.imshow('Resutlt',ImgResult)
       
       
    # cv2.imshow('Thres',ImgThres)
    # cv2.imshow('Warp',ImgWarp)
    # cv2.imshow('Warp Points', ImgWarpPoints)
    # cv2.imshow('Histogram',ImgHist)
    
    return None
        
if __name__ == '__main__':
    cap = cv2.VideoCapture(r'C:\Users\vishe\Downloads\5.avi')
    intialTracbarVals = [0,155,0,185]
    utlis.initializeTrackbars(intialTracbarVals)
    frameCounter=0
    while True:
        frameCounter +=1
        if cap.get(cv2.CAP_PROP_FRAME_COUNT) ==frameCounter:
            cap.set(cv2.CAP_PROP_POS_FRAMES,0)
            frameCounter=0
        
        success, img= cap.read()
        img=cv2.resize(img,(480,240))
        curve=getLaneCurve(img,display=2)
        print(curve)
        #cv2.imshow('Vid',img)
        cv2.waitKey(1) 
