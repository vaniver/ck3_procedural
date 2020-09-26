#!/usr/bin/env python

import os
import random
from imgcat import imgcat
from math import sqrt
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt

from cube import Cube


class Doodler:
    def __init__(self, cubes_colors, size=(500,500), radius=10):
        ''' cubes is a map from (x,y,z) -> [color] '''
        self.im = Image.new('RGB', size, (0, 0, 0))
        self.draw = ImageDraw.Draw(self.im)
        self.size = size
        self.radius = radius
        for cube, color_list in cubes_colors.items():
            if isinstance(color_list, tuple) or len(color_list) == 1:
                color = color_list if isinstance(color_list, tuple) else color_list[0]
                center_pixel = self.get_center(cube)
                hex_points = self.get_hex_points(center_pixel)
                self.draw.polygon(hex_points, fill=color)
            else:
                center_pixel = self.get_center(cube)
                triangle_points = self.get_triangle_points(center_pixel)
                for triangle in range(6):
                    self.draw.polygon(triangle_points[triangle], fill=random.choice(color_list))

    @classmethod
    def random_colors(cls, hex_list, *args, **kwargs):
        random_color = lambda: tuple(np.random.randint(64, 256, size=3))
        cubes_colors = {h: random_color() for h in hex_list}
        return cls(cubes_colors=cubes_colors, *args, **kwargs)

    def get_center(self, cube):
        assert cube.x + cube.y + cube.z == 0, f'bad cube {cube}'
        p = np.array(self.size) / 2
        p += cube.x * np.array([sqrt(3), -1]) * self.radius
        p += cube.y * np.array([0, -1]) * self.radius * 2
        return p

    def get_hex_points(self, p):
        points = []
        for angle in range(0, 360, 60):
            theta = angle * np.pi / 180
            delta = np.array([np.cos(theta), np.sin(theta)]) * self.radius
            points.append(tuple(p + delta))
        return points

    def get_triangle_points(self, p):
        points = []
        for angle in range(0, 360, 60):
            theta = angle * np.pi / 180
            delta = np.array([np.cos(theta), np.sin(theta)]) * self.radius
            points.append(tuple(p + delta))
        points.append(points[0])
        point_sets = []
        for triangle in range(6):
            point_sets.append([tuple(p), points[triangle], points[triangle + 1]])
        return point_sets

    def show(self):
        if os.name == 'posix':
            imgcat(self.im)
        else:
            #self.im.show()
            plt.imshow(self.im)
            plt.show()



if __name__ == '__main__':
    doodler = Doodler(dict([
        (Cube(0,0,0),(255, 255,255)),
        (Cube(1,0,-1),(255,0,0)),
        (Cube(0,-1,1),(0,255,0)),
        (Cube(-1,1,0),(0,0,255)),
    ]))
    doodler.show()
