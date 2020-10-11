#!/usr/bin/env python

import os
import random
from imgcat import imgcat
from math import sqrt
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt

from cube import Cube
from tile import Tile


class BorderDoodler:
    def __init__(self, tile, size=(100,100), radius=10, width=2, depth=None):
        ''' cubes is a map from (x,y,z) -> [color] '''
        self.im = Image.new('RGB', size, (0, 0, 0))
        self.draw = ImageDraw.Draw(self.im)
        self.size = size
        self.radius = radius
        self.width = width
        self.depth = depth
        # First draw all the baronies.
        full_rhl = tile.real_hex_list()
        for cube in full_rhl:
            random_color = tuple(np.random.randint(64, 128, size=3))
            self.draw_barony(cube, random_color)
        # Now recursively draw all of the tile boundaries.
        self.draw_tile_boundary(tile, self.width // 2)
        full_rwl = tile.real_water_list()
        for cube in full_rwl:
            self.draw_barony(cube, (0,0,192))


    def get_center(self, cube):
        assert cube.x + cube.y + cube.z == 0, f'bad cube {cube}'
        p = np.array(self.size) / 2
        p += cube.x * np.array([sqrt(3), -1]) * self.radius
        p += cube.y * np.array([0, -1]) * self.radius * 2
        return p

    def get_hex_points(self, p, r=0):
        points = []
        for angle in range(0, 360, 60):
            theta = angle * np.pi / 180
            delta = np.array([np.cos(theta), -1*np.sin(theta)]) * (self.radius + 1 - r)
            points.append(tuple(p + delta))
        return points

    def draw_barony(self, cube, color):
        """Draw the interior hex for a barony."""
        hex_points = self.get_hex_points(self.get_center(cube), 0)
        self.draw.polygon(hex_points, outline=color, fill=color)

    def draw_boundary(self, cube, rhl, color, scale):
        hex_points = self.get_hex_points(self.get_center(cube), scale)
        hex_points.append(hex_points[0])  # Wraparound.
        for nind, nbor in enumerate(cube.ordered_neighbors()):
            if nbor not in rhl:
                self.draw.line((hex_points[nind], hex_points[nind + 1]), fill=color, width=self.width+1)

    def draw_tile_boundary(self, tile, scale):
        for sub_tile in tile.tile_list:
            if (self.depth is None) or (scale < self.width * self.depth):
                self.draw_tile_boundary(sub_tile, scale + self.width)
        this_rhl = tile.real_hex_list()
        for cube in this_rhl:
            self.draw_boundary(cube, this_rhl, tile.rgb, scale)