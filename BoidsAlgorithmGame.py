# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# # SYSM 6302 Final Project: The Boids Algorithm
# ## [Steve Fox](http://www.stevefox.io) 
# 
# The *Boids Algorithm* is a mathematical model of flocking behavior. The flocking behavior of fish, birds and [stampedes of animals (as in Disney's *The Lion King*)](http://www.lionking.org/movies/Stampede.mov) [4] can be simulated using a simple distributed computational paradigm where each actor in the flock responds only to its local surrounding. The original application of the boids algorithm was for animation (computer graphics), but the literature of output synchronization of systems with localized perception, such as mobile robots, is growing (cite papers). 
# 
# Consider how you, too, may be a boid in a crowd. How do people behave at a concert, or waiting in line for a crowded public event? The average person might try to maintain a small space between him or her and the others, to avoid being jostled around. That person will also try to keep the same speed as those around them: if he slows down too much, you might get trampled; if he goes too fast, he will be jostling the people around him. Further, anybody who is anybody wants to be in the center of the crowd, because the center of the crowd is where anybody who's anybody is. It is simple economics: that's the best place to be (for, perhaps, tangible or intangible reasons). In a natural flock, the center of the flock might be *safer* from predators. Finally, there may be some selfish or globally minded goal-oriented behavior: get to the front of the stage first, or get the best patch of grass when the venue opens.
# 
# Craig W. Reynolds published a paper in July of 1987 titled "Flocks, Herds, and Schools: A Distributed Behavioral Model" describing this behavior for the first time for efficient animation of crowds and flocks. [1] The intent of the paper was to share a computationally feasible method of rendering flocks for computer-generated animations. The computational complexity along with the difficult nature of specifying exact individual paths in large "flocks" (of people, animals, etc...) served as an impetus to explore a more feasible means of simulating the type of behavior exhibited by groups of objects. The essence of the boids algorithm is described very succinctly in [Steve Strogatz's](http://www.stevenstrogatz.com/) [Ted Talk on Syncrhonization in Nature](http://www.ted.com/talks/steven_strogatz_on_sync.html) with footage of flocks of birds and schools of fish found in nature (and majestic music in the background). 
# 
# ## Model of the Boid
# _Boids_ (definition): The word _boid_ is short for _bird-oid_, or a "bird-like object". A _flock_ of boids is one whose emergent behavior displays the characteristics of natural flocks, herds and schools of animals. Simulating these behavioral characteristics is based on three local navigation rules for each _boid_ in the flock:
# 
# ![boid](files/img/boid-red.png "A 2D, directional boid")
# 
# 1. **Collision Avoidance**: Boids (or static or moving obstacles) try not to crash into things
# 
# 1. **Velocity Matching**: Boids try to pre-empt things that try to crash into them, and keep the pace
# 
# 1. **Flock Centering**: Boids try to fly toward the center of the flock

# <codecell>

import sys, pygame
from pygame.locals import *
from math import ceil
from math import cos, sin
from random import randint
# For linear algebra
import numpy as np
import pygame 

from copy import deepcopy # cheap hack; shot myself in the foot with python references in the list

# Model the Boid as an object, as Reynolds suggests in the paper 
# (OOP was not as popular back in 1987; this was originally done in Common LISP)

###############################################################################################################
# Model of the Boid
#   * Could be considered an animal, such as a fish or bird, or a robot
#   * Can only "see" within a small distance (a couple of body lengths), which I call the parameter "neighbor_distance"
#   * The naive complexity is in O(n^2), but it only takes each actor constant time
#       * A massive implementation could use a spatially sorted datastructure such as kD-tree (quad-tree in two dimensions, 
#         octree in three dimensions) for fast approximate nearest neighbors
#   * Resulting behavior is often referred to in the literature as "Emergent Behavior"
#   * If you think the type of dynamical system is interesting, check out cellular automata [5]
#
# Rules
#   * Rule 1: Tries to stay at least "safe_distance" away from its neighbors
#   * Rule 2: Tries to fly toward the center of mass of all neighbors within "neighbor_distance"
#   * Rule 3: Tries to match velocity with the neighbors within neighbor_distance
#
# Coordinate System
#   * The coordinate system in pygame has (0,0) in the top left corner, with "+x" horizontal to the right, and "+y" vertically pointing downward.
###############################################################################################################

def saturate(value, max_value):
    if abs(value) > max_value:
        return np.sign(value)*max_value
    return value

class Boid:
    #: Initialize the boid at some position x,y
    def __init__(self, x, y, angle, image_surface):
        
        #######################################################################################################
        # Dynamical State
        #######################################################################################################
        #: position
        self.pos = np.array([x, y]) # [px] right = +x, down = +y, (0,0) in top left corner
        self.angle = angle # [deg] up = +x = 0[deg], left = +y = 90[deg]; current heading
                
        #: velocity
        # [x, y]
        self.vel = np.array([0.,0.])
        self.rot_vel = 0.0
        
        #: acceleration (this is what the rules affect)
        # [translational, rotational]
        self.accel = np.array([0.0, 0.0])
        self.a2 = np.array([0.,0.])
        
        #######################################################################################################
        # Rendering
        #######################################################################################################
        #: image & rendering surfaces
        self.image_surface = image_surface
        self.surf = pygame.Surface((51,51),flags=pygame.SRCALPHA)
        self.surf.set_alpha(0)
        tmp_rect = self.image_surface.get_rect()
        self.pos_rect = pygame.Rect((16,16,17,17))
        self.surf.blit(image_surface, self.pos_rect)
        
        ############################################
        #: Boid Parameters
        ############################################
        self.max_vel = np.array([25., 20.])
        self.max_accel = np.array([12., 5.]) # magnitude [px] / frame; rotational [deg] / frame
        self.repulsion_coef = 0.5
        self.max_rot_vel = 5
    def __repr__(self):
        return "%.0f,%.0f:%.0f,%.0f " % (self.pos[0], self.pos[1], self.vel[0], self.vel[1]) 
    
    def draw(self):
        #: Rotate the image to the desired heading
        render_angle = -(self.angle+90)
        render_surf = pygame.transform.rotozoom(self.surf, render_angle, 1.0)
        #: Crop the Surface box
        rot_rect = render_surf.get_rect(center=(self.pos[0], self.pos[1]))
        #: Overlay heading 
        #font = pygame.font.Font(None, 12)
        #text = font.render(str(self), 1, (255, 255, 255))
        #r = text.get_rect()
        #render_surf.blit(text, r)
        screen.blit(render_surf, rot_rect)
    
    def update(self, neighbors, close_neighbors, goal, predators=[]):
        #: apply all rules
        self.accel = np.array([0., 0.])

        if len(predators):
            a0 = self.__avoid(predators)
        else:
            a0 = np.array([0.,0.])
        a1 = self.__avoid(close_neighbors)
        a2 = self.__seekgoal(goal)
        a3 = self.__centering(neighbors)
        a4, rot_vel = self.__velocity(neighbors)

        self.accel = 100*a0 

        if np.linalg.norm(self.accel) < self.max_accel[0]:
           new_accel = self.accel + 10*a1
           if np.linalg.norm(self.accel) < self.max_accel[0]:
               self.accel = new_accel
         #+ 0.01*a2 + 0.01*a3 + 0.1*a4

        # Arbitrate acceleration by saturating the acceleration accumulator
        # as in [1]
        if np.linalg.norm(self.accel) < self.max_accel[0]:
           new_accel = self.accel + 0.02*a2
           if np.linalg.norm(self.accel) < self.max_accel[0]:
               self.accel = new_accel
        if np.linalg.norm(self.accel) < self.max_accel[0]:
            new_accel = self.accel + 0.02*a3
            if np.linalg.norm(new_accel) < self.max_accel[0]:
                self.accel = new_accel
        if np.linalg.norm(self.accel) < self.max_accel[0]:
            new_accel = self.accel + 0.02*a4
            if np.linalg.norm(new_accel) < self.max_accel[0]:
                self.accel = new_accel
        
        self.vel = self.accel# + 0.1*self.vel

        # Saturate the Velocity as well
        if np.linalg.norm(self.vel) > self.max_vel[0]:
            self.vel /= np.linalg.norm(self.vel)
            self.vel *= self.max_vel[0]
        
        if np.linalg.norm(self.vel) != 0:
            dir_mvt = np.rad2deg(np.arctan2(self.vel[1],self.vel[0]))
            self.angle = dir_mvt
            self.angle %= 360
       
        self.pos += self.vel

        self.pos[0] %= 1024 # wrap to screen (boids appear on the other side of screen)
        self.pos[1] %= 768 # wrap to screen
    
    ##################################
    # Rules in Order of Precedence
    ##################################

    #: Rule 0: Avoid Predators    
    #: Rule 1: Collision Avoidance
    def __avoid(self, close_neighbors):
        xdot = np.array([0.,0.])
        for i in close_neighbors:
            if i[1] == 0:
                xdot -= i[0].vel
            else:
                xdot -= (i[0].pos-self.pos)/i[1]*(1.0/(i[1])) #*(1.0/i[1]**2) # repulsion is inverse exponentially repulsed by the object
        return xdot

    #: Rule 2: Velocity Matching
    def __velocity(self, neighbors):
        if len(neighbors) == 0:
            return self.vel, self.rot_vel

        sum_vel = np.array([0.,0.])
        sum_rot_vel = 0.

        for i in neighbors:
            sum_vel += i[0].vel
            sum_rot_vel += i[0].rot_vel

        sum_vel /= len(neighbors)
        sum_rot_vel /= len(neighbors)
        
        return sum_vel, sum_rot_vel

    #: Rule 3: Flock Centering
    def __centering(self, neighbors):
        center = np.array([0.,0.])
        
        if len(neighbors) == 0:
            return center
            
        for i in neighbors:
            center += i[0].pos
        
        center /= len(neighbors)
        # compute the direction to the center from current position
        accel_request = (center - self.pos) # vector in pixels
        
        return accel_request

    #: Rule 4: Migratory Urge
    def __seekgoal(self, goal):
        # goal should be np.array([mouse_x, mouse_y])
        accel_request = (goal-self.pos)
        
        #if np.linalg.norm(accel_request) < 500:
        #if np.linalg.norm(accel_request) != 0:
        #    accel_request /= np.linalg.norm(accel_request)
        #else:
        #    accel_request = np.array([0.,0.],dtype=np.float64)
        return accel_request
    
class Flock:
    #: Initializes a flock of @param count boids. Note that birds of a feather
    #: flock together, but that a property of boids is that they can split up
    #: Pseudocode from [3] is adapted here
    def __init__(self, count, safe_distance, neighbor_distance):
        self.image_surface = pygame.image.load("img/boid-red.png").convert_alpha()
        self.safe_distance = safe_distance # [pixels]; 3 * 20 [px] = 3 body lengths
        self.neighbor_distance = neighbor_distance
        self.boid_list = [ Boid(randint(1,1023), randint(1,768), randint(0,360), self.image_surface) for i in range(count) ]
        self.goal = np.array([0.0, 0.0], dtype=np.float64)
        self.mouse_pos = (0,0) #don't use for computation, just rendering
        self.mouse_predator = False
    #: Render the flock
    def draw(self):
        for i in self.boid_list:
            i.draw()
        # draw the goal as a green circle
        pygame.draw.circle(screen, (0,255,0), self.goal.astype(int), 10) # NOTE THAT SCREEN IS A GLOBAL VARIABLE
        # optionally draw the predator as a red circle
        if self.mouse_predator is True:
            pygame.draw.circle(screen, (255,0,0), self.mouse_pos, 10)
        
    def update(self, mouse_pos, mouse_predator):
        static_boid_list = self.boid_list[:] # need to research how to avoid this nonsense
        self.mouse_predator = mouse_predator
        self.mouse_pos = mouse_pos
        for i in self.boid_list:
            # find all neighbors within a radius=safe_distance
            close_neighbors = []
            neighbors = []
            predators = []
            
            if self.mouse_predator:
                predator = Boid(mouse_pos[0], mouse_pos[1], 0,  self.image_surface)
                vector_diff = i.pos - predator.pos
                distance = np.linalg.norm(vector_diff)
                if distance < self.safe_distance:
                    predators.append( (predator, distance) )
            else:
                self.goal[0] = mouse_pos[0]
                self.goal[1] = mouse_pos[1]
            for j in static_boid_list:
                vector_diff = i.pos-j.pos
                distance = np.linalg.norm(vector_diff)
                if j == i:
                    continue
                if distance < self.safe_distance:
                    close_neighbors.append( (j, distance) )
                if distance < self.neighbor_distance:
                    neighbors.append( (j,distance) )

            i.update(neighbors, close_neighbors, self.goal, predators)

# <headingcell level=2>

# Game Loop

# <codecell>

#: Setup the pygame Game Engine [2]
pygame.init()
fpsClock = pygame.time.Clock()

#: Configure the screen size (not resizable)
width = 1024
height = 768
size = (width, height)

black = 1.0, 1.0, 1.0
white = 255,255,255
screen = pygame.display.set_mode(size)
pygame.display.set_caption("Interactive Boid's Algorithm Simulation")

#: Create and render the initial flock
f = Flock(30, 100., 500.) # [number of boids], [avoidance_distance], [neighborhood_distance]
f.draw()
pygame.display.flip()

g_mouse_predator = False

#: Start the simulation
while True:
#    fpsClock.tick(30)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()
        if event.type == pygame.KEYDOWN:
            if event.key == K_ESCAPE:
                pygame.event.post(pygame.event.Event(QUIT))
        if event.type == pygame.MOUSEBUTTONDOWN:
            g_mouse_predator = not g_mouse_predator
    if True:
                screen.fill(white)
                f.update(pygame.mouse.get_pos(), g_mouse_predator)
                f.draw()
                pygame.display.flip()

# <markdowncell>

# # References                                                                                                                                                               
# 
# [1]() Craig W. Reynolds. Flocks, Herds and Schools: A Distributed Behavioral Model. In _ACM SIGGRAPH Computer Graphics_, Volume 21, pp. 25-34. ACM, 1987.
# 
# **Description:** The original boids algorithm paper
# 
# [2](http://www.pygame.org/docs/tut/intro/intro.html) Pete Shinners. Python Pygame Introduction. http://www.pygame.org/docs/tut/intro/intro.html. Accessed 30 November 2013.
# 
# **Description:** Tutorial for using PyGame where I took the basic structure of the code
# 
# [3](http://www.kfish.org/boids/pseudocode.html) Conrad Parker. Boids Pseudocode. http://www.kfish.org/boids/pseudocode.html, 1995. Last Modified 06 September 2007. Accessed 01 December 2013.
# 
# **Description:** Pseudocode for the boids algorithm, with some additional notes and analysis.
# 
# [4](http://www.lionking.org/movies/Stampede.mov) Stampede Sequence from Disney's *The Lion King*, 1995. Available online at: http://www.lionking.org/movies/Stampede.mov. Originally Accessed from http://www.red3d.com/cwr/boids/.                                                                                                                    
#                                                                                                                                                                             
# **Description:** "Quick, Stampede, in the gorge. Simba's down there!"                                                                                                       
#                                                                                                                                                                             
# [5](http://en.wikipedia.org/wiki/Cellular_automaton "Cellular Automata") "Cellular Automata on *Wikipedia.org*. http://en.wikipedia.org/wiki/Cellular_automaton. Accessed 02 December 2013."                                                                                                                                                           
#                                                                                                                                                                             
# **Description:** If you find this type of distributed, dynamical system interesting, you may also be interested in cellular automata.     

