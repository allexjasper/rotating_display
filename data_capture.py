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





#download sensor data from server using a UDP socket
def download_sensor_data_udp():
    serverAddressPort   = ("0.0.0.0", 20001)
    bufferSize          = 256
    UDPServerSocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
    UDPServerSocket.bind(serverAddressPort)
    data = []
    startTimestamp = int(time.time()*1000)
    while len(data) < 60000:
        rawPacket = UDPServerSocket.recvfrom(bufferSize)
        global angle
        global timestamp
        global lock
        match =  re.match("(\d*), (\d*)", rawPacket[0].decode('utf-8'))
        angle =  int(match.group(1)) / 1000.0
        timestamp = int(match.group(2))
        data.append([angle, timestamp, int(time.time()*1000) - startTimestamp])
        #print(
        #        "angle: " + str(angle) + 
        #        " sensor timestamp: " + str(timestamp) + "local timestamp: " + str(time.time())
        #    )

    with open('sample_data_out.csv', 'w') as f:
            for sample in data:
                f.write(str(sample[1]) + ";" + str(int(sample[0]*100)) + ";" + str(sample[2]) + "\n")

download_sensor_data_udp()
