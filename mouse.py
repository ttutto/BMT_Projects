import cv2
import numpy as np
import HandTrackingModule as htm
import time
import autopy

wCam, hCam = 640, 480
cap =cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
pTime = 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0
smoothening = 20
detector = htm.handDetector(maxHands=1)
wScr,hScr = autopy.screen.size()
frameR = 100
print(wScr, hScr)

while True:
    #1. 손의 위치를 마킹하기
    success,img = cap.read()
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)
    #print(lmList)
    # 2. 검지 중지 손가락 정보 얻기
    if len(lmList)!=0:
        x1,y1 =lmList[8][1:]
        x2,y2 =lmList[12][1:]
        #print(x1,y1,x2,y2)
        
        # 3. 어떤손가락을 올렸는지 판단하기
        fingers = detector.fingersUp()
        #print(fingers)
        cv2.rectangle(img,(frameR,frameR),(wCam-frameR, hCam-frameR), (255, 0, 255), 2)
        
        # 4. 검지손가락을 올렸을때 포인터를 움직이기위한 조건문
        if fingers[1]==1 and fingers[2] == 0:
            # 5. 카메라의 해상도와 모니터 화면의 해상도를 변환하기
            
            x3 = np.interp(x1, (frameR, wCam-frameR), (0, wScr))
            y3 = np.interp(y1, (frameR, hCam-frameR), (0, hScr))
            # 6. 마우스 포인터속도를 부드럽게 하기
            clocX = plocX +(x3 - plocX) / smoothening
            clocY = plocY +(y3 - plocY) / smoothening
            
            # 7. 마우스를 움직이는 코드
            autopy.mouse.move(wScr - clocX, clocY)
            cv2.circle(img, (x1,y1), 15, (255, 0, 255), cv2.FILLED)
            plocX,plocY = clocX, clocY
        # 8. 마우스를 클릭하기위한 코드    
        if fingers[1]==1 and fingers[2] == 1:
            length, img, lineInfo = detector.findDistance(8, 12, img)
            print(length)
            if length < 25:
                cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)
                autopy.mouse.click()
            
    
    
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (20,50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    
    cv2.imshow("Image", img)
    if cv2.waitKey(1)&0xFF == ord('q'):
        break
cv2.destroyAllWindows()
