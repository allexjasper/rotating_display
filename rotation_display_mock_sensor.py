# Description: This script is used to mock the sensor data for the rotation display
import socket
import time
import re

bufferSize  = 256
angle = 0
timestamp = 1692470388701
rendererAddress   = ("127.0.0.1", 20001)
UDPClientSocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)

 
# Using readlines()
file1 = open('sample_data.csv', 'r')
Lines = file1.readlines()
 
data = []

count = 0
# Strips the newline character
for line in Lines:
    match =  re.match("(\d*);(\d*.\d*)", line.strip())
    if match == None:
        continue
    tuple = [0,0.0]
    tuple[0] =  float(match.group(2))
    tuple[1] = int(match.group(1))
    data.append(tuple)


dataIndex = 0
while(True):
    #angle = angle + 100
    timestamp = timestamp + 10
    
    angle = data[dataIndex % len(data)][0]
    #timestamp = data[dataIndex % len(data)][1]
    
    payload = str(int(angle * 1000.0)) + ", " + str(timestamp)

    bytesToSend = str.encode(payload)

    UDPClientSocket.sendto(bytesToSend, rendererAddress)
    time.sleep(1/100)
    dataIndex = dataIndex + 1
