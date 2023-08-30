import pygame
import OpenGL
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *

# Author of sensor access related code : Matt Hawkins
# https://www.raspberrypi-spy.co.uk/

import sys
from concurrent import futures
import logging
import flask
from flask import request, jsonify, Response

import threading
import time

import smbus
import time
from ctypes import c_short
from ctypes import c_byte
from ctypes import c_ubyte


lock = threading.Lock()
angle = 0
timestamp = 0

def fetchAngle():
    AS5600_SCL = 19
    AS5600_SDA = 22
    AS5600_ADDRESS = 0x36
    ANGLE_H = 0x0E
    ANGLE_L = 0x0F
    bus = smbus.SMBus(1)
    
    while(True):
        
        buf = bus.read_i2c_block_data(AS5600_ADDRESS, ANGLE_H, 2)
        
        global angle
        global timestamp
        global lock
        lock.acquire()
        angle = (((buf[0]<<8) | buf[1])/4096.0) * 360.0
        #print(time.time())
        timestamp = time.time() * 1000
        #print(angle)
        lock.release()



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

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        lock.acquire()
        local_angle = -angle
        lock.release()
        
        #roate OpenGL display
        glLoadIdentity()
        glOrtho(0, x, y, 0, -10, 10)  
        glTranslatef(x/2, y/2, 0.0)
        glRotatef(local_angle, 0, 0, 1)

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
        pygame.time.wait(10)

srvThread = threading.Thread(target=fetchAngle)
srvThread.start()
box2d()