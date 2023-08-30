import pygame
import OpenGL
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *


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
        #roate OpenGL display
        glLoadIdentity()
        glOrtho(0, x, y, 0, -10, 10)  
        glTranslatef(x/2, y/2, 0.0)
        glRotatef(45.0, 0, 0, 1)

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


box2d()