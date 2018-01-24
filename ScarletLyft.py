
# coding: utf-8

# In[ ]:

# import the necessary packages
from flask import Flask, render_template, Response
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import playsound
import argparse
import imutils
import time
import dlib
import cv2

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')



path = "haar-face.xml"
faceCascade = cv2.CascadeClassifier(path)


# Variable used to hold the ratio of the contour area to the ROI
#global ratio
ratio = 0

# variable used to hold the average time duration of the yawn
global yawnStartTime
yawnStartTime = 0

# Flag for testing the start time of the yawn
global isFirstTime
isFirstTime = True

# List to hold yawn ratio count and timestamp

global yawnRatioCount
yawnRatioCount = []



# Yawn Counter
#global yawnCounter
#yawnCounter = 0

# yawn time
#averageYawnTime = 2.5
averageYawnTime = 2.5

def sound_alarm():
    # play an alarm sound
    playsound.playsound("alarm.wav")
def calculateContours(image, contours):
    cv2.drawContours(image, contours, -1, (0,255,0), 3)
    maxArea = 0
    secondMax = 0
    maxCount = 0
    secondmaxCount = 0
    for i in contours:
        count = i
        area = cv2.contourArea(count)
        if maxArea < area:
            secondMax = maxArea
            maxArea = area
            secondmaxCount = maxCount
            maxCount = count
        elif (secondMax < area):
            secondMax = area
            secondmaxCount = count

    return [secondmaxCount, secondMax]

"""
Thresholds the image and converts it to binary
"""
def thresholdContours(mouthRegion, rectArea):
    global ratio

    # Histogram equalize the image after converting the image from one color space to another
    # Here, converted to greyscale
    imgray = cv2.equalizeHist(cv2.cvtColor(mouthRegion, cv2.COLOR_BGR2GRAY))

    # Thresholding the image => outputs a binary image.
    # Convert each pixel to 255 if that pixel each exceeds 64. Else convert it to 0.
    ret,thresh = cv2.threshold(imgray, 64, 255, cv2.THRESH_BINARY)

    # Finds contours in a binary image
    # Constructs a tree like structure to hold the contours
    # Contouring is done by having the contoured region made by of small rectangles and storing only the end points
    # of the rectangle
    (_,contours, _) = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    returnValue = calculateContours(mouthRegion, contours)

    # returnValue[0] => secondMaxCount
    # returnValue[1] => Area of the contoured region.
    secondMaxCount = returnValue[0]
    contourArea = returnValue[1]

    ratio = contourArea / rectArea

    # Draw contours in the image passed. The contours are stored as vectors in the array.
    # -1 indicates the thickness of the contours. Change if needed.
    if(isinstance(secondMaxCount, np.ndarray) and len(secondMaxCount) > 0):
        cv2.drawContours(mouthRegion, [secondMaxCount], 0, (255,0,0), -1)
def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])

    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)

    # return the eye aspect ratio
    return ear

# construct the argument parse and parse the arguments


# define two constants, one for the eye aspect ratio to indicate
# blink and then a second constant for the number of consecutive
# frames the eye must be below the threshold for to set off the
# alarm
def get_frame():

    EYE_AR_THRESH = 0.3
    flag = 0
    EYE_AR_CONSEC_FRAMES = 20 #48

    # initialize the frame counter as well as a boolean used to
    # indicate if the alarm is going off
    COUNTER = 0
    ALARM_ON = False

    # initialize dlib's face detector (HOG-based) and then create
    # the facial landmark predictor
    print("[INFO] loading facial landmark predictor...")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    # grab the indexes of the facial landmarks for the left and
    # right eye, respectively
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]


    #grab the index for mouth
    (s,e) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]


    # start the video stream thread
    print("[INFO] starting video stream thread...")
    vs = cv2.VideoCapture(0)
    time.sleep(1.0)
    mouthCount=0
    tempm = (s+e)/2.0
    # loop over frames from the video stream

    yawnCounter = 0
    i = 1
    while True:
        # grab the frame from the threaded video file stream, resize
        # it, and convert it to grayscale
        # channels)

        #global ratio, yawnStartTime, isFirstTime, yawnRatioCount, yawnCounter

        r, frame = vs.read()

        imgencode = cv2.imencode('.jpg', frame)[1]
        stringData = imgencode.tostring()
        yield (b'--frame\r\n'
               b'Content-Type: text/plain\r\n\r\n' + stringData + b'\r\n')

        frame = imutils.resize(frame, width=450)
        #blink
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #yawn
        grey = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

        grey = cv2.GaussianBlur(grey, (5, 5), 0)

        # detect faces in the grayscale frame
        rects = detector(gray, 0)
        #rectsss = detector(grey,0)
        faces = faceCascade.detectMultiScale(
                grey,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(50, 50),
                #flags=cv2.cv.CV_HAAR_SCALE_IMAGE
                flags=cv2.CASCADE_SCALE_IMAGE
            )


        # loop over the face detections
        for ((rect), (x, y, w, h)) in zip(rects,faces):
            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy
            # array
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            # extract the left and right eye coordinates, then use the
            # coordinates to compute the eye aspect ratio for both eyes
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)

            # average the eye aspect ratio together for both eyes
            ear = (leftEAR + rightEAR) / 2.0

            # compute the convex hull for the left and right eye, then
            # visualize each of the eyes
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

            # check to see if the eye aspect ratio is below the blink
            # threshold, and if so, increment the blink frame counter
            if ear < EYE_AR_THRESH:
                COUNTER += 1

                # if the eyes were closed for a sufficient number of
                # then sound the alarm
                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                # if the alarm is not on, turn it on
                    if not ALARM_ON:
                        ALARM_ON = True

                        # check to see if an alarm file was supplied,
                        # and if so, start a thread to have the alarm
                        # sound played in the background
                        #t = Thread(target=sound_alarm)
                        #t.deamon = True
                        #t.start()


                    # draw an alarm on the frame
                    cv2.putText(frame, "DROWSINESS ALERT EYES!", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    playsound.playsound("alarm.wav")
                    cv2.putText(frame, "DROWSINESS ALERT EYES!", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # otherwise, the eye aspect ratio is not below the blink
            # threshold, so reset the counter and alarm
            else:
                COUNTER = 0
                ALARM_ON = False

            # Calculate the mouth Yawn
            '''
            mouthHull = cv2.convexHull(mouthP)
            cv2.drawContours(frame, [mouthHull], -1, (0, 200, 0), 1)
            #mouthDim = eye_aspect_ratio(mouthP)

            mouthR = mouthDim/2.0
            if(tempm < mouthR and mouthCount<3):
                mouthCount = mouthCount + 1

            elif(tempm < mouthR and mouthCount>=3):
                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    flag = 1
                # if the alarm is not on, turn it on
                    #if not ALARM_ON:
                        #ALARM_ON = True

                        # check to see if an alarm file was supplied,
                        # and if so, start a thread to have the alarm
                        # sound played in the background

                        # draw an alarm on the frame
                        #cv2.putText(frame, "Mouth Drowsiness", (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                        #if
                        s["alarm"] != "":
                           # t = Thread(target=sound_alarm,
                            #    args=(args["alarm"],))
                            #t.deamon = True
                            #t.start()
                            #break



            '''



        # Draw a rectangle around the faces
        #for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Isolate the ROI as the mouth region
            widthOneCorner = int(x + (w / 4))
            widthOtherCorner = int(x + ((3 * w) / 4))
            heightOneCorner = int(y + (11 * h / 16))
            heightOtherCorner = int(y + h)

        # Indicate the region of interest as the mouth by highlighting it in the window.
            cv2.rectangle(frame, (widthOneCorner, heightOneCorner), (widthOtherCorner, heightOtherCorner),(0,0,255), 2)

            # mouth region
            mouthRegion = frame[heightOneCorner:heightOtherCorner, widthOneCorner:widthOtherCorner]

            # Area of the bottom half of the face rectangle
            rectArea = (w*h)/2

            if(len(mouthRegion) > 0):
                thresholdContours(mouthRegion, rectArea)

            print ("Current probablity of yawn: " + str(round(ratio*1000, 2)) + "%")
            print ("Length of yawncounter: " + str(yawnCounter))

            if(ratio > 0.03):
                if(isFirstTime is True):
                        #isFirstTime = False
                    yawnStartTime = time.time()

                    print(isFirstTime)
                    # If the mouth is open for more than 2.5 seconds, classify it as a yawn
                #if((time.time() - yawnStartTime) >= averageYawnTime):
                    if((yawnStartTime) >= averageYawnTime):
                        yawnCounter += 1
                        print("time if")
                        yawnRatioCount.append(yawnCounter)

                        #if(len(yawnRatioCount) > 6):
                    if(yawnCounter >=6):
                        #isFirstTime = False
                        yawnCounter = 0
                        yawnStartTime = 0
                        #eturnvalue = (True, 'yawn')
                        #f not ALARM_ON:
                        ALARM_ON = True
                        #2= Thread(target=sound_alarm)
                        #2.deamon = True
                        #2.start()



                # draw an alarm on the frame
                        cv2.putText(frame, "DROWSINESS ALERT MOUTH!", (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        playsound.playsound("alarm.wav")

                        cv2.putText(frame, "DROWSINESS ALERT MOUTH!", (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                #eturnvalue = (False, 'yawn')
                    else:
                        ALARM_ON = False

            #lse:
                #eturnvalue = (False,'yawn')

        # Display the resulting frame
        #cv2.namedWindow('yawnVideo')
        #cv2.imshow('yawnVideo', frame)
        #time.sleep(0.025)
        #if cv2.waitKey(1) & 0xFF == ord('q'):
            #sys.exit(0)




        #return False
            # check to see if the eye aspect ratio is below the blink
            # threshold, and if so, increment the blink frame counter

            '''
            elif flag == 2:
                if not ALARM_ON:
                    ALARM_ON = True

                    # check to see if an alarm file was supplied,
                    # and if so, start a thread to have the alarm
                    # sound played in the background
                    if args["alarm"] != "":
                        t = Thread(target=sound_alarm,
                                args=(args["alarm"],))
                        t.deamon = True
                        t.start()

                    # draw an alarm on the frame
                    cv2.putText(frame, "DROWSINESS eyes!", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
              '''
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Isolate the ROI as the mouth region




        # Display the resulting frame
        #cv2.namedWindow('yawnVideo')
        #cv2.imshow('yawnVideo', frame)
        #time.sleep(0.025)
        #if cv2.waitKey(1) & 0xFF == ord('q'):
            #sys.exit(0)
        '''
        if returnvalue[0]:
                if not ALARM_ON:
                        ALARM_ON = True

                        # check to see if an alarm file was supplied,
                        # and if so, start a thread to have the alarm
                        # sound played in the background

                        t = Thread(target=sound_alarm)
                        t.deamon = True
                        t.start()

                # draw an alarm on the frame
                cv2.putText(frame, "DROWSINESS ALERT MOUTH!", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                returnvalue = (False, 'yawn')
        else:
             ALARM_ON = False

                # When everything is done, release the capture
                #yawnCamera.release()

        '''
        # find the value for mouth
        #ptA = tuple(pts-)

            # draw the computed eye aspect ratio on the frame to help
            # with debugging and setting the correct eye aspect ratio
            # thresholds and frame counters
            #cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                #cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # show the frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        i += 1

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()


@app.route('/calc')
def calc():
    return Response(get_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='localhost', debug=True, threaded=True)
