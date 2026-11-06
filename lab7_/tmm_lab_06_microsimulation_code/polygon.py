# Author: Anders Johansson (a.johansson@bristol.ac.uk)
# Description: This file contains functionality to join polygons together
# Date: 2024-10-10

from shapely.geometry import Polygon, Point
from shapely.geometry.multipolygon import MultiPolygon
from shapely.ops import unary_union

def is_inside_polygon(polygon, point):
    if "shapely_polygon" in polygon.keys():
        return Point(point[0], point[1]).within(polygon["shapely_polygon"])
    else:
        polygon_nodes = polygon["nodes"]
        min_x = polygon_nodes[:,0].min()
        min_y = polygon_nodes[:,1].min()
        max_x = polygon_nodes[:,0].max()
        max_y = polygon_nodes[:,1].max()
        if point[0]<min_x or point[0]>max_x or point[1]<min_y or point[1]>max_y:
            return False
        else:
            return True

def join_all_polygons(input_polygons):
    polygons = []
    for poly in input_polygons:
        polygons.append(Polygon(poly["nodes"]))
    # This is needed in order to remove internal polygon boundaries
    if len(polygons)==1:
        polygons.append(polygons[0])
    merged_polygon = unary_union(polygons)
    returned_polygon = []
    boundaries = []
    # Check if polygons are non overlapping
    if type(merged_polygon)==Polygon:
        exterior = merged_polygon.exterior.coords
        n_exterior = len(exterior)
        for i in range(n_exterior-1):
            i2 = (i+1)%n_exterior
            returned_polygon.append([exterior[i][0], exterior[i][1]])
            boundaries.append([exterior[i][0], exterior[i][1], exterior[i2][0], exterior[i2][1]])
    #interior = merged_polygon.interior.coords
    #n_interior = len(interior)
    #for i in range(n_interior-1):
    #    i2 = (i+1)%n_interior
    #    boundaries.append([interior.coords[i][0], interior[i][1], interior[i2][0], interior[i2][1]])

    return returned_polygon, boundaries
