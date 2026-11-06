# Author: Anders Johansson (a.johansson@bristol.ac.uk)
# Description: This file contains functionality to simulate pedestrian crowds.
# Date: 2024-10-10

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import random
import math
import time
from polygon import join_all_polygons, is_inside_polygon
from shapely.geometry import Polygon

def keyboard_event(event):
    if event.key=="escape" or event.key=="q":
        print('You pressed:', event.key, " - bye!")
        exit()

def normalise_vector(vec):
    len = np.linalg.norm(vec)
    if len<0.000001:
        return vec
    else:
        return vec / len
    
def vector_length(vec):
    return np.linalg.norm(vec)

def get_lines_intersection(line1, line2):
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    t = ((x1-x3)*(y3-y4) - (y1-y3)*(x3-x4)) / ((x1-x2)*(y3-y4) - (y1-y2)*(x3-x4))
    u = ((x1-x2)*(y1-y3) - (y1-y2)*(x1-x3)) / ((x1-x2)*(y3-y4) - (y1-y2)*(x3-x4))
    return t, u

def get_nearest_position(pos, boundary):
    x1, y1, x2, y2 = boundary
    tangent = normalise_vector(np.array([x2-x1, y2-y1]))
    normal = np.array([tangent[1], tangent[0]])
    x3, y3, x4, y4 = pos[0], pos[1], pos[0] + normal[0], pos[1] + normal[1]
    t = ((x1-x3)*(y3-y4) - (y1-y3)*(x3-x4)) / ((x1-x2)*(y3-y4) - (y1-y2)*(x3-x4))
    u = ((x1-x2)*(y1-y3) - (y1-y2)*(x1-x3)) / ((x1-x2)*(y3-y4) - (y1-y2)*(x3-x4))
    if t<0:
        t = 0
    elif t>1:
        t = 1
    return np.array([x1 + (x2-x1)*t, y1 + (y2-y1)*t])

class Statistics:
    def __init__(self):
        self.data = {}

    def add(self, keys, value):
        stats = self.data
        n_keys = len(keys)
        for i in range(n_keys):
            key = keys[i]
            if key not in stats:
                if i==n_keys-1:
                    stats[key] = []
                else:
                    stats[key] = {}
            stats = stats[key]
        stats.append(value)
    
    def get(self, keys):
        stats = self.data
        for key in keys:
            if key not in stats:
                return []
            stats = stats[key]
        return stats

class Pedestrian:
    def __init__(self, id, birth_time, starting_position, source, destination, radius, colour, group_name=None):
        self.id = id
        self.birth_time = birth_time
        self.pos = starting_position
        self.source = source
        self.destination = destination 
        self.vel = np.array([0.0, 0.0])
        self.acc = np.array([0.0, 0.0])
        self.desired_direction = np.array([0.0, 0.0])
        self.radius = radius
        self.colour = colour
        self.group_name = group_name
        self.desired_walking_speed = random.gauss(1.34, 0.37)
        self.metadata = {}
        self.previous_polygon = source
        self.previous_polygon_timestamp = birth_time

class World:
    def __init__(self, definition, use_graphics=True):
        random.seed(42)
        self.use_graphics = use_graphics
        self.time = 0.0
        self.definition = definition
        self.polygons = {}
        self.periodic_boundaries = None
        polygons_to_merge = []
        for name, geom in definition["space"].items():
            if geom["type"]=="rectangle":
                x0, y0 = geom["coordinates"][0], geom["coordinates"][1]
                x1, y1 = geom["coordinates"][2], geom["coordinates"][3]
                self.polygons[name] = {
                    "nodes": np.array([[x0,y0], [x0, y1], [x1, y1], [x1, y0]]),
                    "colour": geom["colour"],
                    "collect_statistics": False,
                    "add_boundaries": True
                }
            elif geom["type"]=="polygon":
                self.polygons[name] = {
                    "nodes": np.array(geom["coordinates"]),
                    "colour": geom["colour"],
                    "collect_statistics": False,
                    "add_boundaries": True
                }
            self.polygons[name]["shapely_polygon"] = Polygon(self.polygons[name]["nodes"])
            if "collect_statistics" in geom.keys():
                self.polygons[name]["collect_statistics"] = geom["collect_statistics"]
            if "add_boundaries" in geom.keys():
                self.polygons[name]["add_boundaries"] = geom["add_boundaries"]
            if self.polygons[name]["add_boundaries"]:
                polygons_to_merge.append(self.polygons[name])
        # Define obstacles
        if len(polygons_to_merge)>0:
            self.walkable_space_polygon, self.boundaries = join_all_polygons(polygons_to_merge)
        else:
            self.boundaries = []

        # Add additional boundaries
        if "boundaries" in definition.keys():
            for boundary in definition["boundaries"]:
                self.boundaries.append(boundary)

        # Process pedestrians
        self.pedestrians = {}
        self.next_pedestrian_id = 0
        self.pedestrian_sources = definition["pedestrians"]

        # Process functions
        self.update_desired_directions = None
        self.pedestrian_initialisation = None
        self.process_interactions = None
        for func_name, func_ptr in definition["functions"].items():
            if func_name=="update_directions":
                self.update_desired_directions = func_ptr
            elif func_name=="process_interactions":
                self.process_interactions = func_ptr
            elif func_name=="pedestrian_initialisation":
                self.pedestrian_initialisation = func_ptr
            else:
                raise Exception("Error. Unknown function provided: " + func_name)

        # Process periodic boundaries
        if "periodic_boundaries" in definition.keys():
            self.periodic_boundaries = definition["periodic_boundaries"]

        # Initialise statistics
        self.statistics = Statistics()

        # Calculate window aspect ratio
        border = 2
        self.minX = min([min(b[0], b[2]) for b in self.boundaries]) - border
        self.minY = min([min(b[1], b[3]) for b in self.boundaries]) - border
        self.maxX = max([max(b[0], b[2]) for b in self.boundaries]) + border
        self.maxY = max([max(b[1], b[3]) for b in self.boundaries]) + border
        width = self.maxX - self.minX
        height = self.maxY - self.minY
        if width > height:
            self.maxY = self.maxY * width / height 
        else:
            self.maxX = self.maxX * height / width

        # Show window
        if self.use_graphics:
            self.fig = plt.figure(figsize=(6, 6))
            cid = self.fig.canvas.mpl_connect('key_press_event', keyboard_event)
            plt.ion()

    def __del__(self):
        if self.use_graphics:
            plt.close(self.fig)

    def add_to_statistics(self, keys, value):
        self.statistics.add(keys, value)

    def get_statistics(self, keys=None):
        if keys==None:
            return self.statistics.data
        else:
            return self.statistics.get(keys)

    def get_random_position(self, polygon_name, margin=0.1):
        poly = self.polygons[polygon_name]
        min_x = poly["nodes"][:,0].min()
        min_y = poly["nodes"][:,1].min()
        max_x = poly["nodes"][:,0].max()
        max_y = poly["nodes"][:,1].max()
        x = min_x + margin + (max_x-min_x-2*margin)*random.random()
        y = min_y + margin + (max_y-min_y-2*margin)*random.random()
        return x, y

    def is_inside_polygon(self, pos, polygon_name):
        return is_inside_polygon(self.polygons[polygon_name], pos)

    def update_positions(self, dt=0.05):
        # Update velocities and positions
        self.time += dt
        for i, ped_i in self.pedestrians.items():
            if vector_length(ped_i.acc)>10:
                ped_i.acc = 10*normalise_vector(ped_i.acc)
            ped_i.vel += dt*ped_i.acc
            ped_i.pos += dt*ped_i.vel
        # Apply periodic boundary conditions
        if self.periodic_boundaries != None:
            if self.periodic_boundaries["axis"] == "x":
                for i, ped_i in self.pedestrians.items():
                    if ped_i.pos[0] > self.periodic_boundaries["pos2"]:
                        ped_i.pos[0] = self.periodic_boundaries["pos1"]
                    if ped_i.pos[0] < self.periodic_boundaries["pos1"]:
                        ped_i.pos[0] = self.periodic_boundaries["pos2"]
            elif self.periodic_boundaries["axis"] == "y":
                for i, ped_i in self.pedestrians.items():
                    if ped_i.pos[1] > self.periodic_boundaries["pos2"]:
                        ped_i.pos[1] = self.periodic_boundaries["pos1"]
                    if ped_i.pos[1] < self.periodic_boundaries["pos1"]:
                        ped_i.pos[1] = self.periodic_boundaries["pos2"]

    def update_transitions_statistics(self):
        for i, ped in self.pedestrians.items():
            for poly_name, poly in self.polygons.items():
                if poly["collect_statistics"]:
                    if self.is_inside_polygon(ped.pos, poly_name) and poly_name!=ped.previous_polygon:
                        self.add_to_statistics(["transition_times", ped.previous_polygon, poly_name], self.time - ped.previous_polygon_timestamp)
                        ped.previous_polygon = poly_name
                        ped.previous_polygon_timestamp = self.time

    def birth_pedestrians(self, dt):
        for group_name, data in self.pedestrian_sources.items():
            birth_rate = data["birth_rate"]
            max_count = data["max_count"]
            while len(self.get_statistics(["source", group_name]))<max_count and random.random() < dt*birth_rate:
                radius = 0.4 + 0.1*random.random()
                x0, y0 = self.get_random_position(polygon_name=data["source"], margin=radius)
                ped = Pedestrian(self.next_pedestrian_id, self.time, np.array([x0, y0]), data["source"], data["destination"], radius, data["colour"], group_name=group_name)
                self.pedestrians[self.next_pedestrian_id] = ped
                if self.pedestrian_initialisation != None:
                    self.pedestrian_initialisation(self.pedestrians[self.next_pedestrian_id], self.polygons, self.statistics)
                self.next_pedestrian_id += 1
                self.add_to_statistics(["source", group_name], self.time)

    def remove_pedestrians(self):
        keys_to_remove = []
        for i, ped_i in self.pedestrians.items():
            if self.is_inside_polygon(ped_i.pos, ped_i.destination):
                self.add_to_statistics(["completed_journeys", ped_i.source, ped_i.destination], self.time - ped_i.birth_time)
                keys_to_remove.append(i)
        for key in keys_to_remove:
            self.pedestrians.pop(key, None)

    def render(self):
        if not self.use_graphics:
            return
        plt.cla()
        circle = []
        for i in range(16):
            circle.append([math.cos(2*math.pi*i/16.0), math.sin(2*math.pi*i/16.0)])
        circle = np.array(circle)

        for name, poly in self.polygons.items():
            plt.fill(poly["nodes"][:,0], poly["nodes"][:,1], facecolor=poly["colour"], edgecolor="none")
        # Plot pedestrians
        for ped in self.pedestrians.values():
            x0, y0, rad, col = ped.pos[0], ped.pos[1], ped.radius, ped.colour
            plt.fill(x0+circle[:,0]*rad, y0+circle[:,1]*rad, facecolor=col, edgecolor="none")
        # Plot boundaries
        for boundary in self.boundaries:
            plt.plot([boundary[0], boundary[2]], [boundary[1], boundary[3]], "k-", linewidth=5)

        # Update limits
        plt.xlim(self.minX, self.maxX)
        plt.ylim(self.minY, self.maxY)
        plt.title(str(round(100*self.time)/100.0) + " s")
        plt.xlabel("x (m)")
        plt.ylabel("y (m)")
        plt.show(block=False)
        #plt.pause(0.001)
        plt.pause(0.01)

    def update(self, dt=0.05):
        self.update_desired_directions(self.pedestrians, self.boundaries, self.polygons)
        if self.process_interactions != None:
            self.process_interactions(self.pedestrians, self.boundaries, self.polygons)
        self.update_positions(dt)
        self.update_transitions_statistics()
        self.remove_pedestrians()
        self.birth_pedestrians(dt)
