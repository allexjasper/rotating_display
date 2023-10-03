
import socket
import time
import re
import matplotlib.pyplot as plt
#plt.style.use('seaborn-whitegrid')
import numpy as np
import random
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from scipy import interpolate
from tslearn.clustering import TimeSeriesKMeans
from tslearn.datasets import CachedDatasets
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

from tslearn.generators import random_walks
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn import metrics
import tslearn
import math

sampleRate = 1
 
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

    interpolation_function = interpolate.interp1d(unevenTimestampsNP, unevenAnglesNP, kind='linear')
    evenly_resampled_values = interpolation_function(evenTimestampsNP)

    resampledRotation[0] = evenTimestampsNP
    resampledRotation[1] = evenly_resampled_values
    resampledRotation[2] = rotation[2]
    
    return resampledRotation

#extract matching candiadte ranges from reference rotation matcing withe margin of maxMargin
#also add an additional iteration of the reference rotation if required
def slice_reference_rotation(referenceRotation, subRotation, maxMargin):
    firstSample = None
    if(subRotation[0][0] > maxMargin):
        firstSample = subRotation[0][0] - maxMargin
    else:
        firstSample = 0
    
    lastSample = None
    if(subRotation[0][-1] + maxMargin < referenceRotation[0][-1]):
        lastSample = subRotation[0][-1] + maxMargin
    else:
        lastSample = referenceRotation[0][-1]
    
    slicedReferenceRotation = referenceRotation.copy()
    slicedReferenceRotation[0] = referenceRotation[0][int(firstSample/sampleRate):int(lastSample/sampleRate)]
    slicedReferenceRotation[1] = referenceRotation[1][int(firstSample/sampleRate):int(lastSample/sampleRate)]

    return slicedReferenceRotation
    
   



def find_sub_rotation_lcss(referenceRotation, subRotation):
    np.random.seed(0)
    # Sample longer_time_series (reference time series)
    longer_time_series = np.array(referenceRotation[1])
    # Sample shorter_sub_time_series (sub-time series to match)
    shorter_sub_time_series = np.array(subRotation[1])

    dataset = tslearn.utils.to_time_series_dataset([longer_time_series, shorter_sub_time_series])
    scaler = TimeSeriesScalerMeanVariance(mu=0., std=1.)  # Rescale time series
    dataset_scaled = dataset #scaler.fit_transform(dataset)

    s1 = dataset_scaled[0, :, 0]
    s2 = dataset_scaled[1, :len(subRotation[1]), 0]

    t1 = time.time()
    lcss_path, sim_lcss = metrics.lcss_path(s1, s2, eps=.01)
    print("lcss time: " + str(time.time() - t1))
    #dtw_path, sim_dtw = metrics.dtw_path(s1, s2)

    offset = lcss_path[0][0]

    #plt.figure(1, figsize=(8, 8))

    g1 = dataset_scaled[0, :, 0]
    #plt.plot(g1, "b-", label='First time series')
    g2 = dataset_scaled[1, :len(subRotation[1]), 0]
    #plt.plot(g2, "g-", label='Second time series')

    #for positions in lcss_path:
    #    plt.plot([positions[0], positions[1]],
    #            [dataset_scaled[0, positions[0], 0], dataset_scaled[1, positions[1], 0]], color='orange')
    #plt.legend()
    #plt.title("Time series matching with LCSS")

    #plt.tight_layout()
    #plt.show()
    return offset

def apply_dwt(longer_time_series, shorter_sub_time_series, i):
    long_x_aligned = []
    for j in range (0, len(shorter_sub_time_series)):
        #long_x_aligned.append([j, longer_time_series[i+j][0]])
        long_x_aligned.append(longer_time_series[i+j][0])

    short_x_aligned = []
    for j in range (0, len(shorter_sub_time_series)):
        #short_x_aligned.append([j, shorter_sub_time_series[j][0]])
        short_x_aligned.append(shorter_sub_time_series[j][0])

    return np.linalg.norm(np.array(short_x_aligned) - np.array(long_x_aligned)), None 
    
    #c = np.correlate(long_x_aligned, short_x_aligned)

    #subsequence = longer_time_series[i:i+len(shorter_sub_time_series)]
    #return -c[0], None #fastdtw(long_x_aligned, short_x_aligned, dist=euclidean)


def from_middle_dwt(longer_time_series, shorter_sub_time_series, start, end):
    distance_l, path_s = apply_dwt(longer_time_series, shorter_sub_time_series, start + int(((end - start -10)/2)*0.9))
    distance_r, path_e = apply_dwt(longer_time_series, shorter_sub_time_series, start + int(((end - start - 10)/2)*1.1))
    distance_m, path_m = apply_dwt(longer_time_series, shorter_sub_time_series, start + int((end - start -10)/2))

    d = None
    best_distance = None
    best_position = None
    
    if(distance_r <= distance_l):
        d = 1
        best_distance = distance_r
        best_position = start + int(((end - start -10)/2)*1.1)
    else:
        d = -1
        best_distance = distance_l
        best_position = start + int(((end - start -10)/2)*0.9)

    i = best_position
    while(i > 0 and i < len(longer_time_series) - len(shorter_sub_time_series) + 1):
        distance, path = apply_dwt(longer_time_series, shorter_sub_time_series, i)
        if(distance <= best_distance):
            best_distance = distance
            best_position = i
            i += d
        else:
            break
    
    
    return best_position




def find_sub_rotation_dtw(referenceRotation, subRotation):
    np.random.seed(0)
    # Sample longer_time_series (reference time series)
    #longer_time_series = referenceRotation[1]
    # Sample shorter_sub_time_series (sub-time series to match)
    #shorter_sub_time_series = subRotation[1]

    longer_time_series = []
    for i in range (0, len(referenceRotation[1]) -1):
        longer_time_series.append([referenceRotation[1][i], referenceRotation[0][i]])

    
    shorter_sub_time_series = []
    for i in range (0, len(subRotation[1]) -1):
        shorter_sub_time_series.append([subRotation[1][i], subRotation[0][i]])
    
    # Initialize variables to keep track of the best match
    best_distance = float('inf')
    #best_position = from_middle_dwt(longer_time_series, shorter_sub_time_series, 0, len(longer_time_series) - len(shorter_sub_time_series) + 1 )

    distance_a, path_a = apply_dwt(longer_time_series, shorter_sub_time_series, 130)
    distance_b, path_b = apply_dwt(longer_time_series, shorter_sub_time_series, 200)
    distance_c, path_c = apply_dwt(longer_time_series, shorter_sub_time_series, 0)

    #return 130

    i = 0
    while( i < len(longer_time_series) - len(shorter_sub_time_series) + 1):
        distance, path = apply_dwt(longer_time_series, shorter_sub_time_series, i)
        print("i:", i, "distance:", distance, "best_distance:", best_distance)
        if(distance <= best_distance):
            #print out i, distance, best_distance
            
            
            best_distance = distance
            best_position = i
        i += 1
        


    print("Best Match Position:", best_position)
    print("Best DTW Distance:", best_distance)
    
    return best_position




#test the curve matching my slecting a random rotation and random sub rotation, ploting them and then calling find_sub_rotation
#to match the sub rotation to the refernce rotation
def test_find_sub_rotation(rotations, referceRotation):
    #selRotation = 42 #random.randint(0, len(rotations) - 1)
    selRotation = random.randint(0, len(rotations) - 1)
    rotation = rotations[selRotation]
    subRotation = rotation.copy()
    
    subRotationLength = 0.1 #sub rotation is 10% of the rotation
    #subRotationStart = 0.3 #random.random() * (1 - subRotationLength)
    subRotationStart = random.random() * (1 - subRotationLength)
    subRotation[1] = rotation[1][int( (len(rotation[1]) - 1) * subRotationStart ):
                                 int( (len(rotation[1]) - 1) * (subRotationStart + subRotationLength))] 
    

    resampledSubRotation = resample_rotation(subRotation, 150)

    #offset = find_sub_rotation_lcss(slice_reference_rotation(referceRotation, resampledSubRotation, 200), resampledSubRotation)
    offset = find_sub_rotation_dtw(slice_reference_rotation(referceRotation, resampledSubRotation, 200), resampledSubRotation)


    #plot the refernce rotation and sub rotation into 1 plot
    x = referenceRotation[0] #time stamps
    y = referenceRotation[1] #angles
    sr_x = resampledSubRotation[0] #time stamps
    sr_y = resampledSubRotation[1] #angles
    sr_xm = sr_x.copy()

    for i in range(0, len(sr_x)):
        sr_xm[i] = sr_x[i] + offset * sampleRate -200

        
    plt.plot(x, y, 'o', color='black', alpha=0.5)
    plt.plot(sr_x, sr_y, 'o', color='red', alpha=0.5)
    plt.plot(sr_xm, sr_y, 'o', color='green', alpha=0.5)
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

referenceRotation = resample_rotation(medianRotation)
#print_reference_rotation(referenceRotation)
#print_median_rotation(medianRotation)
test_find_sub_rotation(rotations, referenceRotation)


print("done")