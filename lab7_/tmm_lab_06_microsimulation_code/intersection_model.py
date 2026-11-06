# Author: Anders Johansson (a.johansson@bristol.ac.uk)
# Description: This file contains functionality to simulate intersecting pedestrian and vehicle flows.
# Date: 2024-10-27

from world import World, normalise_vector, vector_length, get_nearest_position
import math
import numpy as np
import random
from matplotlib import pyplot as plt

class Vehicle:
    def __init__(self, x, v, v_desired, time_introduced=0):
        self.x = x                          # Current vehicle position (m)
        self.v = v                          # Current vehicle speed (m/s)
        self.v_desired = v_desired          # Desired (free) vehicle speed (m/s)
        self.time_introduced = time_introduced

class RoadSimulation:
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
        self.completed_journeys = []
        # Model parameters
        self.inflow = 3.0                                # Cars per second
        self.dt = 0.1                                    # Time step (s)
        self.Lc = 5                                      # Car length (m)
        self.Lr = 200                                    # Road length (m)
        self.tau = 1                                     # Relaxation time (s)
        self.d_c = 15                                    # Model parameter, characteristic distance (m)
        # Intersection definition
        self.intersection = {"position": self.Lr / 2, "is_green": True}

    def update(self, dt):
        self.time += dt
        # Introduce new cars onto the road
        if random.random()<self.inflow*dt:
            # Make sure that there is free space
            if len(self.vehicles)==0 or self.vehicles[0].x>self.Lc:
                speed = 10 + 2*random.random()
                self.vehicles.insert(0, Vehicle(x=0, v=0, v_desired=speed, time_introduced=self.time))

        # Update vehicles
        n = len(self.vehicles)
        for i in range(n-1, -1, -1):
            d_alpha = self.vehicles[i+1].x - self.vehicles[i].x if n>1 and i<n-1 else 999999.0
            d_alpha_light = self.intersection["position"] - self.vehicles[i].x if (not self.intersection["is_green"]) and self.vehicles[i].x < self.intersection["position"] else 999999.0
            v_e = self.vehicles[i].v_desired/2*(math.tanh(min(d_alpha, d_alpha_light) - self.d_c) + math.tanh(self.d_c))
            acc = (v_e - self.vehicles[i].v)/self.tau

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
                self.completed_journeys.append(self.time - self.vehicles[i].time_introduced)
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


# Global variables to keep track of the distance to the nearest car at the intersection (and its speed)
nearest_car_distance = 999
nearest_car_speed = 999

def pedestrian_initialisation(pedestrian, polygons, statistics):
    pass

def update_directions(pedestrians, boundaries, polygons):
    for i, ped in pedestrians.items():
        destination_polygon_centroid = polygons[ped.destination]["nodes"].mean(axis=0)
        ped.desired_direction = normalise_vector(destination_polygon_centroid - ped.pos)

def process_interactions(pedestrians, boundaries, polygons):
    # Parameters
    tau = 0.5
    A_boundary_social, B_boundary_social = .5, 2
    A_boundary_physical = 50
    A_social, B_social = .5, 2
    A_physical = 4
    Lambda = 0.2
    for i, ped_i in pedestrians.items():
        ped_i.acc = (ped_i.desired_walking_speed*ped_i.desired_direction - ped_i.vel) * (1/tau)
        # Pairwise forces from other pedestrians
        for j, ped_j in pedestrians.items():
            if i!=j:
                distance = vector_length(ped_i.pos - ped_j.pos)
                tangent = normalise_vector(ped_i.pos - ped_j.pos)
                # Angular dependency
                vec_ij = normalise_vector(ped_j.pos - ped_i.pos)
                cos_phi = normalise_vector(ped_i.vel).dot(vec_ij)
                angular_dependency = Lambda + (1.0 - Lambda)*((1+cos_phi)/2.0)
                # Apply physical force
                if distance <= ped_i.radius+ped_j.radius:
                    ped_i.acc += A_physical*tangent
                # Apply social force
                ped_i.acc += A_social*angular_dependency*math.exp(-(distance)/B_social)*tangent
        # Forces from boundaries
        for boundary in boundaries:
            pos_b  = get_nearest_position(ped_i.pos, boundary)
            distance = vector_length(ped_i.pos - pos_b)
            tangent = normalise_vector(ped_i.pos - pos_b)
            # Apply physical boundary force
            if distance <= ped_i.radius:
                ped_i.acc += A_boundary_physical*tangent
            # Apply social boundary force
            ped_i.acc += A_boundary_social*math.exp(-(distance)/B_boundary_social)*tangent
        # Boundary at traffic light
        time_to_impact = nearest_car_distance / (0.0001 + nearest_car_speed)
        if time_to_impact<5 and nearest_car_distance<50:
            boundary = [0, -3, 1000, -3]
            pos_b  = get_nearest_position(ped_i.pos, boundary)
            distance = vector_length(ped_i.pos - pos_b)
            tangent = normalise_vector(ped_i.pos - pos_b)
            # Apply boundary force
            ped_i.acc += 10*A_boundary_social*math.exp(-(distance)/B_boundary_social)*tangent


# Define modelling scenario
road_length = 200
corridor_width = 10
corridor_length = 10
world_definition = {
    "space": {
        "start": {"type": "rectangle", "coordinates": [road_length/2 - corridor_width/2, -3-corridor_length, road_length/2+corridor_width/2, -3], "colour": "blue", "add_boundaries": False},
        "end": {"type": "rectangle", "coordinates": [road_length/2 - corridor_width/2, 3, road_length/2 + corridor_width/2, 3+2], "colour": "red", "add_boundaries": False},
    },
    "pedestrians": {
        "group1": {"source": "start", "destination": "end", "colour": "red", "birth_rate": .2, "max_count": 9999}
    },
    "boundaries": [[-10, -3-corridor_length, -10, 3+corridor_length], [210, -3-corridor_length, 210, 3+corridor_length]],
    "functions": {
        "update_directions": update_directions,
        "process_interactions": process_interactions,
        "pedestrian_initialisation": pedestrian_initialisation
    }
}

# Set inflow rates of cars and pedestrians
cars_inflow = 0.1   # Cars per second
peds_inflow = 0.1   # Pedestrians per second

# Set up models
world_definition["pedestrians"]["group1"]["birth_rate"] = peds_inflow
world = World(world_definition)
road = RoadSimulation()
road.fig = world.fig
road.inflow = cars_inflow
road.intersection = {"position": 100 - corridor_width/2, "is_green": True}

# Run simulation for 600 seconds
dt = 0.05
for i in range(600*20):
    # Check if pedestrians are crossing road
    peds_currently_crossing = False
    for _, ped_i in world.pedestrians.items():
        if ped_i.pos[1]>=-5 and ped_i.pos[1]<=3:
            peds_currently_crossing = True
    # Check distance to nearest car
    nearest_car_distance = 999999
    nearest_car_speed = 999999
    for car in road.vehicles:
        if car.x <= road.intersection["position"]:
            dist = road.intersection["position"] - car.x
            if dist < nearest_car_distance:
                nearest_car_distance = dist
                nearest_car_speed = car.v
    # Stop cars at intersections in case of pedestrians currently crosing the road
    if peds_currently_crossing:
        road.intersection["is_green"] = False
    else:
        road.intersection["is_green"] = True

    # Update models
    world.update(dt)
    road.update(dt)

    # Render graphics
    if i%20==0:
        road.render()
        world.render(clear_graphics=False)

# Collect flow data
completed_vehicle_journeys = road.completed_journeys
completed_pedestrian_journeys = world.get_statistics(["completed_journeys", "start", "end"])

# Show journey times
print("Vehicle joruney times", completed_vehicle_journeys)
print("Pedestrian joruney times", completed_pedestrian_journeys)
