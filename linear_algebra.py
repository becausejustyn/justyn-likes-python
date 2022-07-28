
import numpy as np
import math

def euclidean_distance(x1, x2):
    '''Calculates the l2 distance between two vectors'''
    distance = 0
    # Squared distance between each coordinate
    for i in range(len(x1)):
        distance += pow((x1[i] - x2[i]), 2)
    return math.sqrt(distance)
    
def euclidean(x1: int, y1: int, x2: int, y2: int):
            return(math.sqrt((y2 - y1)**2 + (x2 - x1)**2))

def manhattan_distance(point1: int, point2: int):
    '''
    Compute the Manhattan distance between two points.
    :param point1 (tuple) representing the coordinates of a point in a plane
    :param point2 (tuple) representing the coordinates of a point in a plane
    :return: (integer)  The Manhattan distance between the two points
   '''
    x1, y1 = point1
    x2, y2 = point2
    distance = abs(x1 - x2) + abs(y1 - y2)
    return(distance)

def closest_point(point1: int, other_points: int):
    '''
    Find the coordinates of the closest point to point1
    :param point1 (tuple) representing the coordinates of a point in a plane
    :param other_points(set) representing several points in a plane
    :return: (tuple) the coordinates of the closest point to point1
    '''
    if not other_points:
        return None
    closest = min(other_points, key=lambda p:manhattan_distance(point1, p))
    return(closest)