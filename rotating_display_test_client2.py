
import socket
import time
import re
import matplotlib.pyplot as plt
import numpy as np
import random
import math
import scipy.interpolate
import threading
import collections
import copy
import statistics

sampleRate = 1

magnetOffset = 100000 #321520 #100000 
 
# Using readlines()
file1 = open('data\sample_data_out3V.csv', 'r')
Lines = file1.readlines()

medianRotation = None

inbound = collections.deque(maxlen=20000) #inbound queue of samlpes (angle, sensor time stamp, local timestamp), 
lock_inbound = threading.Lock()
rotations = collections.deque(maxlen=100)
lock_rotations = threading.Lock()
clock_offset = None #time offset between sensor and local time
transmission_delay = 150
reference_rotation = None #current reference roation
lock_reference_rotation = threading.Lock()
reference_rotation_local_time_pair = [None,None] #reference rotation, sensor time stamp




   
                #current reference roation
                #time offset sensor and local time

def receive_sensor_data_udp():
    serverAddressPort   = ("0.0.0.0", 20001)
    bufferSize          = 256
    UDPServerSocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
    UDPServerSocket.bind(serverAddressPort)
    while(True):
        rawPacket = UDPServerSocket.recvfrom(bufferSize)
        decodedPacket = rawPacket[0].decode('utf-8')
        
        match =  re.match("(\d*), (\d*)", decodedPacket)
        if match == None:
            continue
        angle =  (int(match.group(1)) - magnetOffset)/ 1000.0
        #print(angle) #remove after calibration
        timestamp = int(match.group(2))
        tuple = [0,0.0,0] #angle, sensor time stamp, local timestamp
        tuple[0] = angle
        tuple[1] = timestamp
        tuple[2] = int(time.time() * 1000)
        lock_inbound.acquire()
        inbound.append(tuple)
        lock_inbound.release()               


def read_input_from_CSV(): 
    global inbound
    global lock_inbound
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


def calculate_clock_offset():
    global inbound
    global lock_inbound
    global clock_offset
    
    with lock_inbound:
        # get the last 50 samples from inbound
        last_50_samples = list(inbound)[-50:]

    # calculate the time differences between sensor and local time for each sample
    time_differences = [sample[1] - sample[2] for sample in last_50_samples]

    if len(time_differences) < 10:
        return

    # calculate the median time difference
    median_time_difference = statistics.median(time_differences)

    # the absolute value of the median time difference is the estimated time offset
    clock_offset = int(median_time_difference)

def start_offset_thread():
    # call calculate_offset every 5 minutes
    while(True):
        if(clock_offset is  None):
            calculate_clock_offset()
        time.sleep(5)

#segment data in samplesDeq into rotations
def segment_data(data):
    samples = data
    rotations = []  
    
    prevSample = None;
    curRotation = [0, [], 0]   # start timestamp, samples, duration
    for sample in samples:
        if prevSample is not None:
            
            #print(sample[0])
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

#thread function for data segmentation
def segment_data_thread():
    global rotations
    global inbound
    global lock_inbound
    global lock_rotations
    while(True):
        lock_inbound.acquire()
        samples = copy.deepcopy(inbound)
        lock_inbound.release()
        segment_data(samples)

        lock_rotations.acquire()
        newRotations = segment_data(samples)

        #add new rotations to the global list
        rotations.extend(newRotations)

        print("rotations: " + str(len(rotations)))  
        lock_rotations.release()
        time.sleep(1)

#thread function for calculating the reference rotation
def reference_rotation_thread():
    global reference_rotation
    global lock_reference_rotation
    global rotations
    while(True):
        curRotations = []
        lock_rotations.acquire()
        curRotations.extend(copy.deepcopy(rotations))
        lock_rotations.release()

        if(len(curRotations) < 10):
            time.sleep(5)
            continue

        #find rotation with median duration
        medianRotation = find_rotation_with_median_duration(curRotations)

        #resample rotation
        referenceRotation = resample_rotation(medianRotation)

        #extend reference rotation
        referenceRotation = extend_reference_rotation(referenceRotation, int(len(referenceRotation[1])*0.1))

        #update reference rotation
        lock_reference_rotation.acquire()
        if(reference_rotation is None):
            reference_rotation = referenceRotation
        lock_reference_rotation.release()
        time.sleep(3000)



def find_rotation_with_median_duration(rotations):
    rotations.sort(key=lambda x: x[2])
    return rotations[int(len(rotations)/2)]



def print_median_rotation(rotation):
    x = []
    y = []
    for i in range(0,6):
        for sample in rotation[1]:
            x.append((sample[1] - rotation[0]) + i *(rotation[1][-1][1] - rotation[0]))
            y.append(sample[0])
    plt.plot(x, y, 'o', color='black')
    plt.show()



#compute reference rotation from median rotation
def resample_rotation(rotation, offset = 0):
    resampledRotation = [None, None, None] # time stamp index, angles matching to time stampps, duration

    timestampIndexArray = [] #timestamp, gradient, angle  


    unevenTimestamps = []
    unevenAngles = []
    #add samples to timestamp index array
    for sample in rotation[1]:
        unevenTimestamps.append(sample[1] - rotation[0] - offset)
        unevenAngles.append(sample[0])            

    #evenly resample using numpy
    unevenTimestampsNP = np.array(unevenTimestamps)
    unevenAnglesNP = np.array(unevenAngles)
    evenTimestampsNP = np.arange(unevenTimestampsNP[0], unevenTimestampsNP[-1], sampleRate)

    interpolation_function = scipy.interpolate.interp1d(unevenTimestampsNP, unevenAnglesNP, kind='linear')
    evenly_resampled_values = interpolation_function(evenTimestampsNP)

    resampledRotation[0] = evenTimestampsNP
    resampledRotation[1] = evenly_resampled_values
    resampledRotation[2] = rotation[2]
    
    return resampledRotation
   

def search_match_norm(longer_time_series, shorter_sub_time_series, i):
    long_x_aligned = []
    for j in range (0, len(shorter_sub_time_series)):
        #long_x_aligned.append([j, longer_time_series[i+j][0]])
        long_x_aligned.append(longer_time_series[i+j][0])

    short_x_aligned = []
    for j in range (0, len(shorter_sub_time_series)):
        #short_x_aligned.append([j, shorter_sub_time_series[j][0]])
        short_x_aligned.append(shorter_sub_time_series[j][0])

    return np.linalg.norm(np.array(short_x_aligned) - np.array(long_x_aligned)), None 
    

def locate_sub_rotation(referenceRotation, timeStamps, angles, subRotationStart):

  
    # Initialize variables to keep track of the best match
    best_distance = float('inf')
    best_position = None
    margin = 120 #max margin to search for a match

    
    #search with positive offset
    for offset in range(-margin, margin):
        distance = 0
        curRefAngles = []
        for i in range (0, len(timeStamps)):
            curTimeStampInRef = (offset + timeStamps[i] - subRotationStart)
            curAngleInRef = referenceRotation[1][curTimeStampInRef]
            curRefAngles.append(curAngleInRef)
        distance = np.linalg.norm(np.array(curRefAngles) - np.array(angles))

        # do the printing offset, distance, best_distance
        #print(offset, distance, best_distance)

        if(distance < best_distance):
            best_distance = distance
            best_position = offset

   


    print("Best Match Position:", best_position)
    print("Best DTW Distance:", best_distance)
    
    return best_position


def reverse_find_rotation_start(samples, startSampleIndex):
    for i in range(startSampleIndex, 1, -1):
        if samples[i-1][0] < samples[i][0] and samples[i-1][0] <= 0.0 and samples[i][0] > 0.0:
            return i
    return 0

#thread function to estimate the position in the current rotation
def estimate_position_thread():
    #get the last 30 samples from the inbound queue
    #resample the last 30 samples
    #find the sub rotation in the reference rotation
    #estimate the position in the current rotation
    global reference_rotation
    global lock_reference_rotation
    global inbound
    global lock_inbound
    loopnum = 0
    while(True):
        samples = []
        lock_inbound.acquire()
        samples.extend(copy.deepcopy(inbound))
        lock_inbound.release()

        if(len(samples) < 30):
            time.sleep(1)
            continue




        #resample the last 30 samples
        lastSamples = samples[-30:]

        unevenAngles = []
        unevenTimestamps = []
        for sample in lastSamples:
            unevenAngles.append(sample[0])
            unevenTimestamps.append(sample[1])

        unevenTimestampsNP = np.array(unevenTimestamps)
        unevenAnglesNP = np.array(unevenAngles)
        evenTimestampsNP = np.arange(unevenTimestampsNP[0], unevenTimestampsNP[-1] -1, sampleRate)

        interpolation_function = scipy.interpolate.interp1d(unevenTimestampsNP, unevenAnglesNP, kind='linear')
        evenly_resampled_values = interpolation_function(evenTimestampsNP)
        

        #find the sub rotation in the reference rotation
        reference = None
        lock_reference_rotation.acquire()
        if(reference_rotation is not None):
            reference = copy.deepcopy(reference_rotation)
        lock_reference_rotation.release()
        if(reference is None):
            time.sleep(1)
            continue
        
        rotation_start_index = reverse_find_rotation_start(samples, len(samples) - 1)
        if(len(samples) - rotation_start_index) < 200:
            time.sleep(1)
            continue

        rotation_start_ts = samples[rotation_start_index][1]


        offset = locate_sub_rotation(reference, evenTimestampsNP, evenly_resampled_values, rotation_start_ts)

                
        #todo: continue here correlating to local time
        reference_rotation_local_time_pair[0] = (offset + evenTimestampsNP[-1] - rotation_start_ts)
        #reference_rotation_local_time_pair[1] = evenTimestampsNP[-1]
        reference_rotation_local_time_pair[1] = samples[-1][2] #local time stamp of the last sample

        print("offset: " + str(offset))

        #plot the refernce rotation and sub rotation into 1 plot
        #x = reference[0] #time stamps
        #y = reference[1] #angles
        #sr_x = evenTimestampsNP #time stamps
        #sr_y = evenly_resampled_values #angles
        #sr_xm = sr_x.copy()

        #for i in range(0, len(sr_x)):
        #    sr_xm[i] = sr_x[i] + offset * sampleRate 

        #fig, ax = plt.subplots()
        #plt.plot(x, y, 'o', color='black', alpha=0.5)
        #plt.plot(sr_x, sr_y, 'o', color='red', alpha=0.5)
        #plt.plot(sr_xm, sr_y, 'o', color='green', alpha=0.5)
        #plt.savefig('match' + str(loopnum) + '.png')
        time.sleep(5)
        loopnum += 1


#test the curve matching my slecting a random rotation and random sub rotation, ploting them and then calling find_sub_rotation
#to match the sub rotation to the refernce rotation
def test_find_sub_rotation(rotations, referceRotation):
    #selRotation = 42 #random.randint(0, len(rotations) - 1)
    selRotation = random.randint(0, len(rotations) - 1)
    rotation = rotations[selRotation]
    subRotation = rotation.copy()
    
    subRotationLength = 0.1 #sub rotation is 10% of the rotation
    #subRotationStart = 0.3 #random.random() * (1 - subRotationLength)
    subRotationStart = 0.99 #random.random() #* (1 - subRotationLength)
    subRotationStartIndex = int( (len(rotation[1]) - 1) * subRotationStart )
    subRotationEndIndex = int( (len(rotation[1]) - 1) * (subRotationStart + subRotationLength))
    subRotation[1] = rotation[1][subRotationStartIndex:subRotationEndIndex] 
    

    resampledSubRotation = resample_rotation(subRotation, 0)

    t1 = time.time()
    offset = locate_sub_rotation(referceRotation, resampledSubRotation[0], resampledSubRotation[1], 0)
    t2 = time.time()
    print("time: " + str(t2 - t1))

    #plot the refernce rotation and sub rotation into 1 plot
    x = referenceRotation[0] #time stamps
    y = referenceRotation[1] #angles
    sr_x = resampledSubRotation[0] #time stamps
    sr_y = resampledSubRotation[1] #angles
    sr_xm = sr_x.copy()

    for i in range(0, len(sr_x)):
        sr_xm[i] = sr_x[i] + offset * sampleRate 

        
    plt.plot(x, y, 'o', color='black', alpha=0.5)
    plt.plot(sr_x, sr_y, 'o', color='red', alpha=0.5)
    plt.plot(sr_xm, sr_y, 'o', color='green', alpha=0.5)
    plt.show()

#append the first n samples of the ration again to the back to model the wrap arround
def extend_reference_rotation(referenceRotation, n):
    extendedReferenceRotation = referenceRotation.copy()
    extendedReferenceRotation[1] = np.append(extendedReferenceRotation[1], extendedReferenceRotation[1][0:n])
    extendedReferenceRotation[0] = np.arange(0, len(extendedReferenceRotation[1]), sampleRate)
    return extendedReferenceRotation


    

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


lock_reder = threading.Lock()
render_x = []
render_raw = []
render_ref = []

def render_thread():
    latestSample = None

    while(True):
        lock_inbound.acquire()
        
        if len(inbound) > 1:
            latestSample = inbound[-1]
        lock_inbound.release()

        if latestSample is None or reference_rotation_local_time_pair[1] is None or clock_offset is None:
            time.sleep(1)
            continue

        #project the sample 10ms into the future using the reference rotation
        
        timeDiff = latestSample[2] - reference_rotation_local_time_pair[1] + clock_offset
        lock_reference_rotation.acquire()
        if reference_rotation is not None:
            ind = (reference_rotation_local_time_pair[0] + timeDiff + transmission_delay)% reference_rotation[2]
            angle = reference_rotation[1][ind]
        lock_reference_rotation.release()

        lock_reder.acquire()
        render_x.append(latestSample[2])
        render_raw.append(latestSample[0])
        render_ref.append(angle)
        lock_reder.release()
        time.sleep(0.1)



def plot_thread():
    global render_x
    global render_raw
    global render_ref
    i = 0
    while(True):
        fig, ax = plt.subplots()
        lock_reder.acquire()
        x = render_x.copy()
        print( len(x))
        raw = render_raw.copy()
        ref = render_ref.copy()
        render_x.clear()
        render_raw.clear()
        render_ref.clear()
        lock_reder.release()
        plt.plot(x, raw, '-', color='red')
        plt.plot(x, ref, '-', color='green')
        plt.savefig('plot' + str(i) + '.png')
        

        #plt.show()
        time.sleep(1)
        i += 1


#data = read_input_from_CSV()
#rotations = segment_data(data)
#print("rotations: " + str(len(rotations)))
#medianRotation = find_rotation_with_median_duration(rotations)
#print("median rotation duration: " + str(medianRotation[2]))


#referenceRotation = resample_rotation(medianRotation)
#referenceRotation = extend_reference_rotation(referenceRotation, int(len(referenceRotation[1])*0.1))


srvThread1 = threading.Thread(target=receive_sensor_data_udp)
srvThread1.start()
srvThread2 = threading.Thread(target=segment_data_thread)
srvThread2.start()
srvThread3 = threading.Thread(target=reference_rotation_thread)
srvThread3.start()
srvThread4 = threading.Thread(target=estimate_position_thread)
srvThread4.start()
srvThread5 = threading.Thread(target=start_offset_thread)
srvThread5.start()
srvThread5 = threading.Thread(target=render_thread)
srvThread5.start()
#srvThread6 = threading.Thread(target=plot_thread)
#srvThread6.start()

#print_reference_rotation(referenceRotation)
#print_median_rotation(medianRotation)
#test_find_sub_rotation(rotations, referenceRotation)

plot_thread()


#sleep 30 sec
time.sleep(300)

print("done")