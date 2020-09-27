#!/usr/bin/env python

from itertools import product
import random

import numpy as np

from cube import Cube
from doodle import Doodler
from typing import List, Tuple
from dataclasses import dataclass, field

@dataclass
class Tile:
    origin: Cube = field(default_factory=Cube)
    rotation: int = 0
    hex_list: List[Cube] = field(default_factory=lambda: [Cube()])
    tile_list: List["Tile"] = field(default_factory=list)
    water_list: List[Cube] = field(default_factory=list)
    rgb: Tuple[int, int, int] = field(default_factory=lambda: (255, 255, 255))

    @property
    def size(self):
        return len(self.hex_list) + sum(t.size for t in self.tile_list)

    def add_tile(self, other):
        if isinstance(other, Tile):
            self.tile_list.append(other)
        else:
            raise ValueError('Not a tile in tile.add_tile: ' + str(other))

    def add_hex(self, other):
        '''Adds another tile, not checking for collisions.'''
        if isinstance(other, Tile):
            for index, other_el in enumerate(other.hex_list):
                self.hex_list.append(other.origin.add(other_el.rotate_right(other.rotation-self.rotation)))
            for other_el in enumerate(other.water_list):
                self.water_list.append(other.origin.add(other_el.rotate_right(other.rotation-self.rotation)))
        elif isinstance(other, Cube):
            self.hex_list.append(other)
        else:
            raise ValueError('Not a cube or tile in tile.add_hexes: ' + str(other))

    def add_hex_safe(self, other):
        '''Adds another tile, ignoring duplicates.'''
        if isinstance(other, Tile):
            for index, other_el in enumerate(other.hex_list):
                new_hex = other.origin.add(other_el.rotate_right(other.rotation-self.rotation))
                if new_hex not in self.hex_list:
                    self.hex_list.append(new_hex)
        elif isinstance(other, Cube):
            if other not in self.hex_list:
                self.hex_list.append(other)
        else:
            raise ValueError('Not a cube or tile in tile.add_hexes_safe: ' + str(other))

    def add_water(self, other):
        '''Adds a water hex.'''
        if isinstance(other, Cube):
            if other not in self.water_list:
                self.water_list.append(other)
        else:
            raise ValueError('Not a cube in tile.add_water: ' + str(other))

    def real_hex_list(self, duplicates=False):
        '''Adds the origin and rotation to all hexes to get the real hex list represented by the tile.'''
        r_list = []
        if self.origin == Cube(0,0,0) and self.rotation == 0:
            r_list += self.hex_list
            for tile in self.tile_list:
                for l_el in tile.real_hex_list():
                    if duplicates or l_el not in r_list:
                        r_list.append(l_el)
        else:
            r_list = []
            for el in self.hex_list:
                r_list.append(self.origin.add(el.rotate_right(self.rotation)))
            for tile in self.tile_list:
                for el in tile.real_hex_list():
                    l_el = self.origin.add(el.rotate_right(self.rotation))
                    if duplicates or l_el not in r_list:
                        r_list.append(l_el)
        return r_list

    def relative_hex_list(self):
        '''Expands the tile_list to get hexes relative to the origin.'''
        h_list = []
        h_list += self.hex_list
        for tile in self.tile_list:
            for l_el in tile.real_hex_list():
                if l_el not in h_list:
                    h_list.append(l_el)
        return h_list

    def real_water_list(self):
        '''Adds the origin and rotation to all water hexes to get the real hex list represented by the tile.'''
        w_list = []
        for el in self.water_list:
            w_list.append(self.origin.add(el.rotate_right(self.rotation)))
        for tile in self.tile_list:
            for el in tile.real_water_list():
                w_el = self.origin.add(el.rotate_right(self.rotation))
                if w_el not in w_list:
                    w_list.append(w_el)
        return w_list

    def relative_water_list(self):
        '''Expands the tile_list to get water hexes relative to the origin.'''
        w_list = []
        w_list += [el for el in self.water_list]
        for tile in self.tile_list:
            for el in tile.real_water_list():
                if el not in w_list:
                    w_list.append(el)
        return w_list

    def real_total_list(self):
        '''Get all hexes, water or land.'''
        h_list = []
        h_list += self.real_hex_list()
        h_list += self.real_water_list()
        return h_list

    def relative_total_list(self):
        '''Get all hexes, water or land, relative to the origin.'''
        h_list = []
        h_list += self.relative_hex_list()
        h_list += self.relative_water_list()
        return h_list

    # def __str__(self):
    #     result = "[" + str(self.origin)+"; " + str(self.rotation)+"; "
    #     for el in self.hex_list:
    #         result = result + " " + str(el)
    #     for tile in self.tile_list:
    #         result = result + " " + str(tile)
    #     if len(self.water_list)>0:
    #         result = result + "; "
    #         for el in self.water_list:
    #             result = result+" " + str(el)
    #     result = result+"]"
    #     return result

    def rectify(self, recursive=True):
        '''Push the origin and rotation out to the hex_list, restoring (0,0,0) and 0. Precursor to flipping. Untested for hierarchical tiles.'''
        for index, el in enumerate(self.hex_list):
            self.hex_list[index] = self.origin.add(el.rotate_right(self.rotation))
        for index, el in enumerate(self.water_list):
            self.water_list[index] = self.origin.add(el.rotate_right(self.rotation))
        for tile in self.tile_list:
            tile.origin = self.origin.add(tile.origin.rotate_right(self.rotation))
            tile.rotation = (self.rotation + tile.rotation) % 6
            if recursive:
                tile.rectify(recursive)
        self.origin = Cube(0,0,0)
        self.rotation = 0

    def neighbors(self):
        '''All possible *land* neighbors.'''
        neighbors = set()
        self_real = self.real_hex_list()
        for el in self_real:
            neighbors.update(el.neighbors())
        return neighbors - set(self_real) - set(self.water_list)

    def inclusive_neighbors(self):
        '''All possible *land* neighbors or hexes in the tile.'''
        neighbors = set()
        self_real = self.real_hex_list()
        for el in self_real:
            neighbors.update(el.neighbors())
        return neighbors - set(self.water_list)

    def weighted_neighbors(self):
        '''Each neighboring *land* hex will be present for each time it's a neighbor.'''
        neighbors = []
        self_real = self.real_hex_list()
        self_water_real = self.real_water_list()
        for rel in self_real:
            neighbors.extend([el for el in rel.neighbors() if (el not in self_real) and (el not in self_water_real)])
        return neighbors

    def weighted_valid_neighbors(self, other):
        '''Each neighboring *land* hex will be present for each time it's a neighbor, so long as it's valid according to valid.'''
        return [el for el in self.weighted_neighbors() if el.valid(other)]

    def relative_neighbors(self, weighted=True):
        '''All possible land neighbors, present ever time it's a neighbor, in the local reference frame.'''
        neighbors = []
        self_relative = self.relative_hex_list()
        self_water_relative = self.relative_water_list()
        for rel in self_relative:
            neighbors.extend([el for el in rel.neighbors() if (el not in self_relative) and (el not in self_water_relative)])
        return neighbors

    def strait_neighbors(self):
        '''All possible land neighbors that are only reachable by strait.'''
        neighbors = set()
        self_real = self.real_hex_list()
        for el in self_real:
            neighbors.update(el.strait_neighbors())
        return neighbors - set(self_real) - set(self.water_list) - set(self.neighbors())

    def water_neighbors(self):
        '''All possible *water* neighbors.'''
        neighbors = set()
        self_water_real = self.real_water_list()
        for el in self_water_real:
            neighbors.update(el.neighbors())
        return neighbors - set(self.real_hex_list()) - set(self_water_real)

    def weighted_water_neighbors(self):
        '''Each neighboring *water* hex will be present for each time it's a neighbor.'''
        neighbors = []
        for rel in self.real_water_list():
            neighbors.extend([el for el in rel.neighbors() if (el not in self.hex_list) and (el not in self.water_list)])
        return neighbors

    def collision(self, other):
        '''Determines whether this tile collides with another Tile or Cube in absolute position (according to local parents).'''
        if isinstance(other, Tile):
            other_real = other.real_hex_list()
            other_water_real = other.real_water_list()
            self_real = self.real_hex_list()
            self_water_real = self.real_water_list()
            for other_el in other_real:
                if other_el in self_real or other_el in self_water_real:
                    return True
            for other_el in other_water_real:
                if other_el in self_real:
                    return True
            return False
        if isinstance(other, Cube):
            return (other in self.real_hex_list or other in self.real_water_list)
        raise ValueError('Can only detect collisions with Tiles and Cubes: ' + str(other))

    def touching(self, other):
        '''Does other have any land hexes in self's boundary?'''
        # TODO: maybe deprecate
        if isinstance(other, Tile):
            other_real = other.real_hex_list()
            self_neighbors = self.neighbors()
            for other_el in other_real:
                if other_el in self_neighbors:
                    return True
            return False
        elif isinstance(other, Cube):
            return other in self.neighbors()
        else:
            raise TypeError('Not a Tile or Cube in touching(): ' + str(other))


    def connected(self, other):
        ''' Return True if self and other are connected (touching or overlapping) '''
        inclusive_boundary = self.inclusive_neighbors()
        if isinstance(other, Tile):
            other_real = other.real_hex_list()
            for other_el in other_real:
                if other_el in inclusive_boundary:
                    return True
            return False
        if isinstance(other, Cube):
            return other in inclusive_boundary

    def boundary(self):
        '''Each hex is present only if it's on the edge of the tile.'''
        boundary = []
        self_real = self.real_hex_list()
        self_water_real = self.real_water_list()
        for el in self_real:
            if any([(h not in self_real and h not in self_water_real) for h in el.neighbors()]):
                boundary.append(el)
        return boundary

    def relative_boundary(self):
        '''Each hex is present only if it's on the edge of the tile.'''
        boundary = []
        self_hexes = []
        self_water = []
        for el in self.hex_list:
            self_hexes.append(el)
        for el in self.water_list:
            self_water.append(el)
        for tile in self.tile_list:
            self_hexes.extend(tile.real_hex_list())
            self_water.extend(tile.real_water_list())
        for el in self_hexes:
            if any([(h not in self_hexes and h not in self_water) for h in el.neighbors()]):
                boundary.append(el)
        return boundary

    def weighted_boundary(self):
        '''Each hex is present only if it's on the edge of the tile, and is weighted based on how many outside neighbors it has.'''
        boundary = []
        for el in self.hex_list:
            boundary.extend([el]*sum([(h not in self.hex_list and h not in self.water_list) for h in el.neighbors()]))
        return boundary

    @classmethod
    def new_tile(cls, size, *args, **kwargs):
        tile = cls(*args, **kwargs)  # Do what the constructor do
        while len(tile.hex_list) < size:
            tile.add_hex(random.choice(tile.relative_neighbors()))
        return tile

    def add_new_tile(self, size, rgb=(255,255,255), weighted=True, cant=[]):
        """Adds new hexes until it's at size, giving up in 100 tries if impossible.
        Operates in relative space, so cant probably won't do the right thing unless origin is 0,0,0."""
        self_relative = self.relative_hex_list() + self.relative_water_list()
        self_neighbors = self.relative_neighbors(weighted)
        for _ in range(100): #Give up if you can't do it in 100 tries.
            new_origin = random.choice(self_neighbors)
            new_tile = Tile(hex_list=[new_origin], rgb=rgb)
            while len(new_tile.hex_list) < size:
                new_neighbors = new_tile.relative_neighbors(weighted)
                new_neighbors = [x for x in new_neighbors if x not in self_relative and x not in cant]
                if len(new_neighbors) > 0:
                    new_tile.add_hex(random.choice(new_neighbors))
                else:
                    break
            if len(new_tile.hex_list) == size:
                self.tile_list.append(new_tile)
                return
        raise ValueError("Failed to add new tile to {}".format(self))

    def move_into_place(self, must, cant_land, cant_water, weighted=True, max_tries=100, valid_dir=None, debug=False):
        '''Modify origin and rotation such that each set of potential locations in must is occupied, while none of the locations in cant are occupied. (Water hexes can overlap with those in cant_water.)
        must is a list of lists of Cubes, and cant_land and cant_water should be flat lists of Cubes. valid_dir specifies a third (and so all cubes are checked for validity).
        If successful, returns True. If not successful after max_tries attempts, will return False.'''
        if weighted:
            s_n = self.weighted_boundary()
        else:
            s_n = self.relative_boundary()
        merge_found = False
        tries = 0
        for index, must_req in enumerate(must): #We might as well remove impossible requirements, and fail immediately if they can't be met.
            must_req = [el for el in must_req if el not in cant_land and el not in cant_water]
        for must_req in enumerate(must):
            if len(must_req) == 0:
                return False
        while (not merge_found) and tries < max_tries:
            tries += 1
            self.rotation = random.randint(0,5)
            s_edge = random.choice(s_n)
            try:
                must_guaranteed = random.choice(random.choice(must))
            except:
                return False
            self.origin = must_guaranteed.sub(s_edge.rotate_right(self.rotation))
            r_list = self.real_hex_list()
            r_w_list = self.real_water_list()
            if all([any([el in req for el in r_list]) for req in must]):
                valid = True
                if valid_dir:
                    for el in r_list:
                        if not el.valid(valid_dir):
                            valid = False
                if valid:
                    for el in r_list:
                        if el in cant_land or el in cant_water:
                            valid = False
                if valid:
                    for el in r_w_list:
                        if el in cant_land:
                            valid = False
                if valid:
                    merge_found = True
        if debug:
            print(str(tries))
        return merge_found

    def min_dist(self, other):
        ''' Get the minimum distance to another tile (0 if overlapping) '''
        closest = float('inf')
        for hex_a, hex_b in product(self.real_hex_list(), other.real_hex_list()):
            closest = min(closest, hex_a.dist(hex_b))
            if closest <= 0:
                break
        return closest

    def doodle(self, color=(255, 255, 255), show=True):
        doodle = Doodler({h: color for h in self.real_hex_list()})
        if show:
            doodle.show()
        return doodle

    def doodle_by_tile(self, color_list=[(255, 255, 255)], water_color=(0, 128, 255), show=True, size=(500,500)):
        rgb_from_cube = {}
        t_idx = 0
        threshold = list(np.cumsum([len(tile.real_hex_list()) for tile in self.tile_list]))
        for cube_idx, h in enumerate(self.real_hex_list(duplicates=True)):
            if h in rgb_from_cube:
                rgb_from_cube[h].append(color_list[t_idx])
            else:
                rgb_from_cube[h] = [color_list[t_idx]]
            if cube_idx + 1 >= threshold[t_idx]:
                t_idx += 1
        for el in self.real_water_list():
            if el in rgb_from_cube:
                rgb_from_cube[el].append(water_color)
            else:
                rgb_from_cube[el] = [water_color]
        doodle = Doodler(rgb_from_cube, size=size)
        if show:
            doodle.show()
        return doodle


if __name__ == '__main__':
    tile = Tile()
    print(tile)
    tile.doodle(show=True)
    d = {tile: 0}
    print(d)
