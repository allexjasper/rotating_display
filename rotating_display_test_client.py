import pygame
import OpenGL
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import requests
import re
import threading
import time
import socket
import collections
import statistics

lock = threading.Lock()
samplesLock = threading.Lock()
angle = 0
timestamp = 0

samplesDeq = collections.deque(maxlen=10000)
rotations = collections.deque(maxlen=100)
referenceRotation = None

#extract the 95% percentile of the given numbers
def remove_outliers(data):
    return data[int(len(data) * 0.05) : int(len(data) * 0.95)]


def interpolate(start, end, timestampArray):
    #compute gradient
    gradient =  (timestampArray[end%len(timestampArray)][2] - timestampArray[start%len(timestampArray)][2] ) / (end - start)

    #interpolate
    i = start
    while i < end:
        timestampArray[i%len(timestampArray)][2] = timestampArray[start%len(timestampArray)][2] + (i - start) * gradient
        i += 1
            

#averages all angels from the ratoations and computes the reference rotation
def calc_reference_rotation():
    rotationsCopy = rotations.copy()

    rotationDurations = []
    allAngles = []
    for rotation in rotationsCopy:
        #print("duration: " + str(rotation[1][-1][1] - rotation[0])) 
        rotationDurations.append(rotation[1][-1][1] - rotation[0]) #last sample timestamp - first sample timestamp 
        for sample in rotation[1]:
            allAngles.append(sample[0])

    if len(rotationDurations) == 0:
        return
    medianRotationDuration = statistics.median(rotationDurations)
    print("median rotation duration: " + str(medianRotationDuration))


    allAngles.sort()
    allAngles = remove_outliers(allAngles)
    allAngles.reverse()
    allAngles = remove_outliers(allAngles)
    allAngles.reverse()

    #print the angles range
    print("min angle: " + str(allAngles[0]) + " max angle: " + str(allAngles[-1]))
    
    minAngle = allAngles[0]
    maxAngle = allAngles[-1]


    #iterate from the min angle to the max angle and find the angle uinsg 1/100 of a degree
    #anglesIndexArray = []
    #for i in range(int(minAngle * 100), int(maxAngle * 100)):
    #    anglesIndexArray.append([i,  1, []])    #angle, gradient, [samples]
    #    anglesIndexArray.append([i, -1, []])    #angle, gradient, [samples]

    #iterate over all timestamps 1ms at a time and add the angles to the timesampIndexAarray
    timestampIndexArray = []
    for i in range(0, int(medianRotationDuration) + 1):
        timestampIndexArray.append([i, []])    #timestamp, [angles]
    
    
    #index by timestamp and angle
    for rotation in rotationsCopy:
        for sample in rotation[1]:
            #if not (sample[0] < minAngle or sample[0] >= maxAngle):
            #    anglesIndexArray[int(sample[0] * 100) - int(minAngle * 100)] = [int(sample[0] * 100), []] #add angle to the angle index array
            #    anglesIndexArray[int(sample[0] * 100) - int(minAngle * 100)][1].append(sample[1] - rotation[0]) #add sample timestamp to the angle index array

            #todo: fix problem that timestamps are aboslute and relative to start of sample
            if not (sample[1] - rotation[0] < 0 or sample[1] - rotation[0] > medianRotationDuration):
                #timestampIndexArray[int(sample[1]) - rotation[0]] = [sample[1] - rotation[0], []] #add timestamp to the timestamp index array
                timestampIndexArray[int(sample[1] - rotation[0])][1].append(sample[0]) #add sample angle to the timestamp index array

    #compute median out indexed data
    #for angle in anglesIndexArray:
    #    medianAngle = None
    #    if len(angle[1]) > 0:
    #        medianAngle = statistics.median(angle[1])
    #    angle.append(medianAngle)

    for timestamp in timestampIndexArray:
        medianTimestamp = None
        if len(timestamp[1]) > 0:
            medianTimestamp = statistics.median(timestamp[1])
        timestamp.append(medianTimestamp)


    startInterpolationInterval = 0
    endInterpolationInterval = 0

    while startInterpolationInterval < len(timestampIndexArray):
        if timestampIndexArray[startInterpolationInterval][2] is not None:
            startInterpolationInterval += 1
            continue
        else:
            endInterpolationInterval = startInterpolationInterval + 1
            while endInterpolationInterval % len(timestampIndexArray) != startInterpolationInterval % len(timestampIndexArray):
                if timestampIndexArray[endInterpolationInterval % len(timestampIndexArray)][2] is not None:
                    break
                endInterpolationInterval += 1

        #do interpolation
        interpolate(startInterpolationInterval -1, endInterpolationInterval, timestampIndexArray)



        startInterpolationInterval += 1 

    global referenceRotation
    global samplesLock
    samplesLock.acquire()
    referenceRotation = timestampIndexArray    
    samplesLock.release()



    print("timestamp index array length: " + str(len(timestampIndexArray)))




    

#segment data in samplesDeq into rotations
def segment_data():
    samplesLock.acquire()
    samples = samplesDeq.copy()
    samplesLock.release()

    prevSample = None;
    curRotation = [0, []]   # start timestamp, samples
    for sample in samples:
        if prevSample is not None:
            #start of rotation is angle increasing and travsing 0, save crrent rotation and start a new one
            if prevSample[0] < sample[0] and prevSample[0] <= 0.0 and sample[0] >= 0.0:
                #is a fiull rotation?
                if abs(curRotation[1][-1][0]) < 0.5 and abs(curRotation[1][0][0]) < 0.5: #start and end angle are close to 0
                    rotations.append(curRotation.copy())
                curRotation = [0, []]

                curRotation[0] = sample[1]
                curRotation[1].clear()
            curRotation[1].append(sample)
        prevSample = sample


#download sensor data from server using a UDP socket
def download_sensor_data_udp():
    serverAddressPort   = ("0.0.0.0", 20001)
    bufferSize          = 256
    UDPServerSocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
    UDPServerSocket.bind(serverAddressPort)
    while(True):
        rawPacket = UDPServerSocket.recvfrom(bufferSize)
        global angle
        global timestamp
        global lock
        match =  re.match("(\d*), (\d*)", rawPacket[0].decode('utf-8'))
        lock.acquire()
        angle =  int(match.group(1)) / 1000.0
        timestamp = int(match.group(2))
        lock.release()
        #print(
        #        "angle: " + str(angle) + 
        #        " sensor timestamp: " + str(timestamp) + "local timestamp: " + str(time.time())
        #    )

            

correctionAngle = 160
correctionFactor = 1.0
timeCorrection = 7 #ms


#render a 2d box using OpenGL
def box2d():
    pygame.init()
    display = (0,0)
    screen = pygame.display.set_mode(display,  pygame.DOUBLEBUF | pygame.OPENGL)
    x, y = screen.get_size()
    glClearColor(0.0, 0.0, 0.0, 0.0)
    glViewport(0, 0, x, y)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glOrtho(0, x, y, 0, -10, 10)  
    glTranslatef(x/2, y/2, 0.0)

    prevAngle = None;
    rotationStartTimeStamp = None

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        global angle
        global timestamp
        global lock

        lock.acquire()
        local_angle = (-angle +correctionAngle) * correctionFactor 
        local_timestamp = timestamp
        lock.release()
        #print(local_angle)
        samplesLock.acquire()
        samplesDeq.append([local_angle, local_timestamp])   
        samplesLock.release()


        if prevAngle is not None and prevAngle < local_angle and prevAngle <= 0.0 and local_angle >= 0.0:
            rotationStartTimeStamp = int(time.time() * 1000)
            #print("rotation start: " + str(local_timestamp))
        prevAngle = local_angle

        timeCorrectedAngle = local_angle
        samplesLock.acquire()
        if referenceRotation is not None:
            timeStampInex = int(time.time() * 1000) - rotationStartTimeStamp
            print(str( timeStampInex))
            if timeStampInex < len(referenceRotation):
                timeCorrectedAngle = referenceRotation[timeStampInex][2] 
        samplesLock.release()

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        #roate OpenGL display
        glLoadIdentity()
        glOrtho(0, x, y, 0, -10, 10)  
        glTranslatef(x/2, y/2, 0.0)
        glRotatef(timeCorrectedAngle, 0, 0, 1)

        glBegin(GL_QUADS) 
        glColor3f(1, 0, 0)
        glVertex2f(-600, -400)         
        glVertex2f(-600, 400)
        glVertex2f(-10, 400)
        glVertex2f(-10, -400)     
        glEnd()

        glBegin(GL_QUADS) 
        glColor3f(1, 0, 0)
        glVertex2f(10, -400)         
        glVertex2f(10, 400)
        glVertex2f(600, 400)
        glVertex2f(600, -400)     
        glEnd()


        pygame.display.flip()
        #print(str(time.time()))
        pygame.time.wait(10)

def segmentation_thread():
    while(True):
        time.sleep(1)
        segment_data()
        calc_reference_rotation()
        

srvThread = threading.Thread(target=download_sensor_data_udp)
srvThread.start()

srvThread = threading.Thread(target=segmentation_thread)
srvThread.start()

box2d()