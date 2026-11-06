# Author: Anders Johansson (a.johansson@bristol.ac.uk)
# Description: This file contains functionality to simulate pedestrian crowds.
# Date: 2024-10-10

from matplotlib import pyplot as plt
import random
import math
import numpy as np

from polygon import join_all_polygons
def keyboard_event(event):
    if event.key=='escape' or event.key=="q":
        print('You pressed:', event.key, " - bye!")
        exit()

# Show window
fig = plt.figure(figsize=(6, 6))
cid = fig.canvas.mpl_connect('key_press_event', keyboard_event)
plt.ion()

class Vehicle:
    def __init__(self, x, v, v_desired):
        self.x = x                          # Current vehicle position (m)
        self.v = v                          # Current vehicle speed (m/s)
        self.v_desired = v_desired          # Desired (free) vehicle speed (m/s

class Simulation:
    def __init__(self):
        self.time = 0
        # Vehicle data
        self.vehicles = []                               # List of vehicles currently on the road
        self.trajectories = {                            # Vehicle trajectories
            "t": [], 
            "x": [], 
            "density": [],
            "flow": []
        }                                 
        # Model parameters
        self.inflow = 3.0                                # Cars per second
        self.dt = 0.1                                    # Time step (s)
        self.Lc = 5                                      # Car length (m)
        self.Lr = 1000                                   # Road length (m)
        self.tau = 1                                     # Relaxation time (s)
        self.d_c = 50                                    # Model parameter, characteristic distance (m)

    def update(self, dt):
        self.time += dt
        # Introduce new cars onto the road
        if random.random()<self.inflow*dt:
            # Make sure that there is free space
            if len(self.vehicles)==0 or self.vehicles[0].x>self.Lc:
                speed = 30 + 5*random.random()
                self.vehicles.insert(0, Vehicle(x=0, v=0, v_desired=speed))

        # Update vehicles
        n = len(self.vehicles)
        for i in range(n-1, -1, -1):
            d_alpha = self.vehicles[i+1].x - self.vehicles[i].x if n>1 and i<n-1 else 999999.0
            v_e = self.vehicles[i].v_desired/2*(math.tanh(d_alpha - self.d_c) + math.tanh(self.d_c))
            acc = (v_e - self.vehicles[i].v)/self.tau

            # Add traffic jam
            #if self.vehicles[i].x>=700 and self.vehicles[i].x<=710:
            #    self.vehicles[i].v = min(5, self.vehicles[i].v)

            self.vehicles[i].v += dt*acc
            self.vehicles[i].x += dt*self.vehicles[i].v
            # Prevent crashes
            if i<n-1 and self.vehicles[i].x > self.vehicles[i+1].x - 0.01:
                self.vehicles[i].x = self.vehicles[i+1].x - 0.01
            # Add current position to trajectories
            self.trajectories["t"].append(self.time)
            self.trajectories["x"].append(self.vehicles[i].x)
            self.trajectories["density"].append(len(self.vehicles) / self.Lr)
            self.trajectories["flow"].append(len(self.vehicles) / self.Lr * self.vehicles[i].v)

        # Remove vehicles outside road
        for i in range(n):
            if self.vehicles[i].x > self.Lr:
                self.vehicles.pop(i)

    def render(self):
        # Plot current state
        plt.cla()
        plt.fill([0, self.Lr, self.Lr, 0], [-3, -3, 3, 3], facecolor="gray", edgecolor="gray")
        for i in range(len(self.vehicles)):
            plt.plot(self.vehicles[i].x, 0.0, "ko", linewidth=5)

        plt.xlim(-5, self.Lr + 5)
        plt.ylim(-5, self.Lr/4)
        plt.title(str(round(100*self.time)/100.0) + " s")
        plt.show(block=False)
        plt.pause(0.001)


dt = 0.05
sim = Simulation()
for i in range(int(120/dt)):
    sim.update(dt)
    if i%20==0:
        sim.render()
