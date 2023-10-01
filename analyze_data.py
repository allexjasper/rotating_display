
import socket
import time
import re
import matplotlib.pyplot as plt
#plt.style.use('seaborn-whitegrid')
import numpy as np
import random
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean


 
# Using readlines()
file1 = open('data\sample_data_out3V.csv', 'r')
Lines = file1.readlines()

def read_input(): 
    data = []
    count = 0
    # Strips the newline character
    for line in Lines:
        #match =  re.match("(\d*);(\d*.\d*)", line.strip())
        #match =  re.match("(\d*);(\d*)", line.strip())
        match =  re.match("(\d*);(\d*);(\d*)", line.strip())
    
        if match == None:
            continue
        tuple = [0,0.0,0] #angle, sensor time stamp, local timestamp
        tuple[1] =  int(match.group(1))
        tuple[0] = int(match.group(2))/100 - 32152/100
        tuple[2] = int(match.group(3))
        data.append(tuple)
    return data

#segment data in samplesDeq into rotations
def segment_data(data):
    samples = data
    rotations = []  
    
    prevSample = None;
    curRotation = [0, [], 0]   # start timestamp, samples, duration
    for sample in samples:
        if prevSample is not None:
            #start of rotation is angle increasing and travsing 0, save crrent rotation and start a new one
            if prevSample[0] < sample[0] and prevSample[0] <= 0.0 and sample[0] > 0.0:
                #is a fiull rotation?
                if abs(curRotation[1][-1][0]) < 0.5 and abs(curRotation[1][0][0]) < 0.5: #start and end angle are close to 0
                    curRotation[2] = curRotation[1][-1][1] - curRotation[0] #duration
                    rotations.append(curRotation.copy())
                curRotation = [0, [], 0]

                curRotation[0] = sample[1]
                curRotation[1].clear()
            curRotation[1].append(sample)
        prevSample = sample
    return rotations


def find_rotation_with_median_duration(rotations):
    rotations.sort(key=lambda x: x[2])
    return rotations[int(len(rotations)/2)]


data = read_input()
rotations = segment_data(data)
print("rotations: " + str(len(rotations)))
medianRotation = find_rotation_with_median_duration(rotations)
print("median rotation duration: " + str(medianRotation[2]))

def print_median_rotation(rotation):
    x = []
    y = []
    for i in range(0,6):
        for sample in rotation[1]:
            x.append((sample[1] - rotation[0]) + i *(rotation[1][-1][1] - rotation[0]))
            y.append(sample[0])
    plt.plot(x, y, 'o', color='black')
    plt.show()


def interpolate(start, end, timestampArray):
    #compute gradient
    gradient =  (timestampArray[end%len(timestampArray)][2] - timestampArray[start%len(timestampArray)][2] ) / (end - start)

    #interpolate
    i = start
    while i < end:
        timestampArray[i%len(timestampArray)][2] = timestampArray[start%len(timestampArray)][2] + (i - start) * gradient
        i += 1

#compute reference rotation from median rotation
#create a evenly sampled rotation with the angles sampled at .001deg
def compute_reference_rotation(rotation):
    referenceRotation = [None, None, None, None, None] # time index, angle index, duration, max angle, min angle
    
    angleSample = [0, 0, 0] #angle (0.01 deg), gradient, relative sensor time stamp (ms)

    #compute max and min angle from median rotation
    maxAngle = 360
    minAngle = -360
    for sample in rotation[1]:
        if sample[0] > maxAngle:
            maxAngle = sample[0]
        if sample[0] < minAngle:
            minAngle = sample[0]

    referenceRotation[3] = maxAngle
    referenceRotation[4] = minAngle


    timestampIndexArray = [] #timestamp, gradient, angle  

    #initialize timestamp index array
    for i in range(0, int(rotation[2]) + 1):
        timestampIndexArray.append([i, None, None]) #add timestamp to the timestamp index array with empty angle and gradient

    #add samples to timestamp index array
    for sample in rotation[1]:    
        timestampIndexArray[int(sample[1] - rotation[0])][2] = sample[0] #add sample angle to the timestamp index array

    #interpolate missing angles
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
        interpolate(startInterpolationInterval -1, endInterpolationInterval, timestampIndexArray)


    #compute gradient
    for i in range(0, len(timestampIndexArray)):
        if timestampIndexArray[i][2] is not None:
            if i > 0:
                timestampIndexArray[i][1] = (timestampIndexArray[i][2] > timestampIndexArray[i-1][2])
            else:
                timestampIndexArray[i][1] = (timestampIndexArray[i][2] > timestampIndexArray[-1][2])

    referenceRotation[0] = timestampIndexArray
    return referenceRotation







#find sub rotation in the reference rotation that matches the rotation using dynamic time warping, not  using the gradient
def find_sub_rotation(referenceRotation, subRotation):
    
    #prepare fastdtw compliant time series out of reference rotation with (time, angle) tuples
    referenceRotationAngles = []
    for sample in referenceRotation[0]:
        referenceRotationAngles.append([sample[0], sample[2]])

    #prepare fastdtw compliant time series out of sub rotation with (time, angle) tuples
    subRotationAngles = []
    for sample in subRotation[1]:
        subRotationAngles.append([sample[1] - subRotation[0], sample[0]])



    #compute the dynamic time warping distance between the sub rotation and the reference rotation
    distance, path = fastdtw(referenceRotationAngles, subRotationAngles, dist=euclidean)

    #derive out of the return from fastdtw the time offset between the sub rotation and the reference rotation
    timeOffset = path[-1][0] - path[0][0]




    #print the distance and the path
    print("distance: " + str(distance))
    print("path: " + str(path))   


#test the curve matching my slecting a random rotation and random sub rotation, ploting them and then calling find_sub_rotation
#to match the sub rotation to the refernce rotation
def test_find_sub_rotation(rotations, referceRotation):
    rotation = rotations[random.randint(0, len(rotations) - 1)]
    subRotation = rotation.copy()
    
    #randomly choose a float between 0 and 0.9
    subRotationLength = 0.1 #sub rotation is 10% of the rotation
    subRotationStart = random.random() * (1 - subRotationLength)
    
    #subRotation[1] = rotation[1][random.randint(0, int(len(rotation[1])*0.3) - 1):random.randint(int(len(rotation[1]) * 0.4), len(rotation[1]) - 1)]
    subRotation[1] = rotation[1][int( (len(rotation[1]) - 1) * subRotationStart ):
                                 int( (len(rotation[1]) - 1) * (subRotationStart + subRotationLength))] 

    find_sub_rotation(referceRotation, subRotation)
    
    #plot the refernce rotation and sub rotation into 1 plot
    x = []
    y = []
    sr_x = []
    sr_y = []

    for sample in referceRotation[0]:
        x.append(sample[0])
        y.append(sample[2])

    for sample in subRotation[1]:
        sr_x.append(sample[1] - rotation[0] - 300)
        sr_y.append(sample[0])
        
    plt.plot(x, y, 'o', color='black', alpha=0.5)
    plt.plot(sr_x, sr_y, 'o', color='red', alpha=0.5)
    plt.show()



#plot reference rotation
def print_reference_rotation(referenceRotation):
    x = []
    y = []
    g = []
    for sample in referenceRotation[0]:
        x.append(sample[0])
        g.append(sample[1])
        y.append(sample[2])
    plt.plot(x, y, 'o', color='blue', alpha=0.5)
    plt.plot(x, g, 'o', color='red', alpha=0.5)
    plt.show()


#plot data from each rotation as a line in a single plot
def print_all_rotations(rotations):
    count = 0
    for rotation in rotations:
        x = []
        y = []
        for sample in rotation[1]:
            x.append(sample[1] - rotation[0])
            y.append(sample[0])
        plt.plot(x, y, '-', color=(random.random(), random.random(), random.random()))
        count += 1
        if count >= 100:
            break
    plt.show()

referenceRotation = compute_reference_rotation(medianRotation)
#print_reference_rotation(referenceRotation)
#print_median_rotation(medianRotation)
test_find_sub_rotation(rotations, referenceRotation)


print("done")