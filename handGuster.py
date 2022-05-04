import cv2
print(cv2.__version__)
import numpy as np
import time

class mpHands:
    import mediapipe as mp
    # default constructor method to self execute
    def __init__(self,maxHands=1,tol1=.5,tol2=.5):
        self.hands=self.mp.solutions.hands.Hands(False,maxHands,tol1,tol2)
    # marking the hands to calculate the distances.
    def Marks(self,frame):
        myHands=[]
        frameRGB=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        results=self.hands.process(frameRGB)
        if results.multi_hand_landmarks != None:
            for handLandMarks in results.multi_hand_landmarks:
                myHand=[]
                for landMark in handLandMarks.landmark:
                     myHand.append((int(landMark.x*width),int(landMark.y*height)))
                myHands.append(myHand)
        return myHands
# finding the distances between the fingers and key points.
def findDistances(handData):
    distMatrix=np.zeros([len(handData),len(handData)],dtype='float')
    for row in range(0,len(handData)):
        for column in range(0,len(handData)):
            distMatrix[row][column]=((handData[row][0]-handData[column][0])**2+(handData[row][1]-handData[column][1])**2) **(1./2.)
    return distMatrix
# trying to minimise the error.
def findError(gestureMatrix,unknownMatrix,keyPoints):
    error=0
    for row in keyPoints:
        for column in keyPoints:
            error=error+abs(gestureMatrix[row][column]-unknownMatrix[row][column])
    return error            
# finding the distances trying to calculating the gesture by probability.
def findGesture(unknownGesture,knownGestures,keyPoints,gestureNames,tol):
    errorArray = []
    for i in range(0,len(gestureNames),1):
        error=findError(knownGestures[i],unknownGesture,keyPoints)
        errorArray.append(error)
    errorMin = errorArray[0] 
    minIndex=0
    for i in range(0,len(errorArray),1):
        if errorArray[i]<errorMin:
            errorMin=errorArray[i]
            minIndex = i
    if errorMin < tol:
        gesture =gestureNames[minIndex]
    if errorMin >= tol:
        gesture="Unknown"
    return gesture

width=1280
height=720
cam=cv2.VideoCapture (0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT,height)
cam.set(cv2.CAP_PROP_FPS, 30)
cam.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*'MJPG'))
findHands=mpHands(1)
time.sleep(5)
# important parts of the hands
keyPoints=[0,4,5,9,13,17,8,12,16,20]
train=True
tol = 1500 # If the error is less than the tolerance, then we'll say we haveamatch, otherwise we don't.
trainCount = 0
numGesture = int(input('How many gestures do you want to train on? '))
gestureNames = []
knownGestures = []

# creating the set of data for training.
for i in range(1,numGesture+1,1):
   prompt = 'Name of Gesture #'+str(i)+' '
   name=input(prompt)
   gestureNames.append(name)
print(gestureNames)

while True:
    ignore, frame = cam.read()
    frame=cv2.resize(frame,(width,height))
    handData=findHands.Marks(frame)
    # reading the data for trainig.
    if train==True:
        if handData!=[]:
            print('Please gesture ',gestureNames[trainCount],': Press "t" when ready')
            if cv2.waitKey(1) & 0xff==ord('t'):
                print("1")
                knownGesture=findDistances(handData[0])
                knownGestures.append(knownGesture)
                trainCount = trainCount+1
                if trainCount==numGesture:
                    train=False    
    #working model of the gesture detection .                  
    if train == False:
        if handData!=[]:
            unknownGesture=findDistances(handData[0])
            myGesture = findGesture(unknownGesture, knownGestures, keyPoints, gestureNames, tol)
            #error=findError(knownGesture, unknownGesture, keyPoints)
            # writing text onto the frame
            cv2.putText(frame, myGesture,(100,175),cv2.FONT_HERSHEY_SIMPLEX, 3, (255,0,0),8)
    for hand in handData:
        for ind in keyPoints:
            cv2.circle(frame,hand[ind],25,(255,0,255),3)
    cv2.imshow('my WEBcam', frame)
    cv2.moveWindow('my WEBcam',0,0)
    if cv2.waitKey(1) & 0xff ==ord('q'):
        break
cam.release()
        