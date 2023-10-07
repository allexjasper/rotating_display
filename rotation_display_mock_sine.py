# Description: This script is used to mock the sensor data for the rotation display
import socket
import time
import re
import numpy as np
import matplotlib.pyplot as plt
import time

bufferSize  = 256
angle = 0
timestamp = 1692470388701
rendererAddress   = ("127.0.0.1", 20001)
UDPClientSocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)

 
cycle_duration = 6  # seconds
min_angle = -15    # degrees
max_angle = 17     # degrees
total_duration = 600  # seconds (10 minutes)

# Generate time values at a desired sampling rate for each cycle
sampling_rate = 100  # samples per second
num_samples_per_cycle = cycle_duration * sampling_rate

# Calculate the number of cycles needed to reach the total duration
num_cycles = int(total_duration / cycle_duration)

# Generate the sine curve for angle values (radians) for each cycle
sine_curve = np.sin(2 * np.pi * np.linspace(0, 1, num_samples_per_cycle))

# Create an empty array to store the concatenated angle values
angle_values = np.array([])

# Generate the sequence by concatenating multiple cycles
for _ in range(num_cycles):
    cycle_angles = (sine_curve + 1) * (max_angle - min_angle) / 2 + min_angle
    angle_values = np.concatenate((angle_values, cycle_angles))

# Generate time values for the entire sequence
time_values = np.linspace(0, total_duration, len(angle_values))

# Visualization (optional)
#plt.figure(figsize=(10, 4))
#plt.plot(time_values, angle_values)
#plt.xlabel('Time (s)')
#plt.ylabel('Angle (degrees)')
#plt.title('Sine Wave of Angles')
#plt.grid(True)
#plt.show()



dataIndex = 0
while(True):
    #angle = angle + 100
    timestamp = time.time() * 1000 -20 #time_values[dataIndex % len(time_values)]
    
    angle = angle_values[dataIndex % len(time_values)] + 100
    #timestamp = data[dataIndex % len(data)][1]
    
    payload = str(int(angle * 1000)) + ", " + str(int(timestamp))

    bytesToSend = str.encode(payload)

    UDPClientSocket.sendto(bytesToSend, rendererAddress)
    time.sleep(1/100)
    dataIndex = dataIndex + 1
