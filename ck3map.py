from cube import Cube
from dataclasses import dataclass
from enum import Enum
import heapq
from itertools import combinations, count
from math import sqrt
import PIL.Image
import os
import random
from typing import Tuple

import numpy as np

def dist2line(x, y, srcx, srcy, destx, desty):
    return abs((desty-srcy)*x-(destx-srcx)*y+destx*srcy-desty*srcx)/sqrt((desty-srcy)**2+(destx-srcx)**2)

Terrain = Enum('Terrain','plains farmlands hills mountains desert desert_mountains oasis jungle forest taiga wetlands steppe floodplains drylands')

TERRAIN_HEIGHT = {Terrain.farmlands: (0,1), Terrain.plains: (0,2), Terrain.floodplains: (0,2), Terrain.taiga: (0,2),
                  Terrain.wetlands: (0,2), Terrain.steppe: (0,2), Terrain.drylands: (0,2),
                  Terrain.oasis: (0,3), Terrain.desert: (0,3),
                  Terrain.jungle: (1,5), Terrain.forest: (1,5),
                  Terrain.hills: (5,20), 
                  Terrain.mountains: (15,55),  Terrain.desert_mountains: (15,55), }

# HEIGHTMAP constants
WATER_HEIGHT = 96

# PROVINCES constants
IMPASSABLE = (0, 0, 0)

# RIVERS constants
MAJOR_RIVER_THRESHOLD = 9
RIVER_EXTEND = 5
SOURCE = 0
MERGE = 1
SPLIT = 2
WATER = 254
LAND = 255

@dataclass
class RiverEdge:
    width: int = 0
    start_trio: Tuple[int, int, int] = (0, 0, 0)
    start_xy: Tuple[int, int] = (0, 0)
    end_trio: Tuple[int, int, int] = (0, 0, 0)
    end_xy: Tuple[int, int] = (0, 0)
    merge: bool = False
    source: bool = False
    name: str = "River"


def new_rgb(dictionary):
    '''Returns a randomly chosen RGB value that isn't already in the values of dictionary.'''
    guess = tuple(np.random.randint(0,256,3))
    if guess in dictionary.values() or guess == (255,255,255) or guess == (0,0,0):
        return new_rgb(dictionary)
    else:
        return guess


class CK3Map:
    
    def __init__(self, max_x=8192, max_y=4096, hex_size=40, map_size=40, crisp=True, default=None):
        '''Creates a map of size max_x x max_y, with hexes that have radius hex_size pixels, and at most map_size hexes on an edge. If crisp=True (default), the hexes will all be regular and the same size; if crisp=False, the sizes will vary to make them visually distinct, and will extend to the edge of the image boundary.'''
        self.max_x = max_x
        self.max_y = max_y
        self.mid_x = max_x/2
        self.mid_y = max_y/2
        self.hex_size = hex_size
        self.default = default
        mr = int(min(map_size, max_x/(hex_size*4), max_y/(hex_size*4)))
        self.map_rad = mr
        self.num_hexes = 3*(self.map_rad*(self.map_rad+1))+1
        self.scale = max(1,64/mr)
        self.img_provinces = PIL.Image.new('RGB', (max_x,max_y),  "white")
        self.img_topology = PIL.Image.new('L', (max_x,max_y),  "white")
        self.img_rivers = PIL.Image.new('P', (max_x,max_y),  255)  # 255=white
        with open(os.path.join("data", "rivers_palette.txt"), "r") as f:
            self.img_rivers.putpalette([int(s) for s in f.read().split(',')])
        self.d_cube2rgb = {}
        self.d_cube2terr = {}
        self.d_cube2pid = {}
        self.d_pid2cube = {}
        self.d_trio2vertex = {}
        self.d_trio2river = {}
        self.s3 = sqrt(3)
        self.l_valid_cubes = None
        self.crisp = crisp
        self.cx = np.zeros(self.num_hexes)
        self.cy = np.zeros(self.num_hexes)
        self.c2cx = {}
        self.c2cy = {}
        for ind, c in enumerate(self.valid_cubes):
            i = c.x
            j = c.y
            k = c.z
            self.cx[ind] = int(i * 1.5 * self.hex_size + self.mid_x)
            self.cy[ind] = self.max_y - int((j * 3 * self.hex_size + self.cx[ind] - self.mid_x) / self.s3 + self.mid_y)
            self.c2cx[c] = self.cx[ind]
            self.c2cy[c] = self.cy[ind]
        if self.crisp:
            self.paint_edge = self.paint_edge_crisp
            self.paint_major_river = self.paint_major_river_crisp
        if not self.crisp:
            self.paint_edge = self.paint_edge_non_crisp
            self.paint_major_river = self.paint_major_river_non_crisp
            self.sigma = np.zeros(self.num_hexes)
            self.c2sigma = {}
            for ind, c in enumerate(self.valid_cubes):
                self.sigma[ind] = np.random.rand() + 0.5
                self.c2sigma[c] = self.sigma[ind]

    def pixel_to_cube(self, x,y):
        if self.crisp:
            q = (x-self.mid_x) * 2.0/3.0/self.hex_size
            r = (-(x-self.mid_x) / 3.0 + self.s3/3.0 * (y-self.mid_y))/self.hex_size
            return self.cube_round(q,-q-r,r)
        return self.valid_cubes[(((self.cx-x)**2+(self.cy-y)**2)/self.sigma).argmin()]
        
    def cube_round(self, i, j, k):
        ri, rj, rk = int(round(i)), int(round(j)), int(round(k))
        di, dj, dk = abs(ri-i), abs(rj-j), abs(rk-k)

        if di > dj and di > dk:
            ri = -rj-rk
        elif dj > dk:
            rj = -ri-rk
        else:
            rk = -ri-rj
        return Cube(ri, rj, rk)

    def cube_to_pid(self, i, j=None, k=None):
        '''Note that province id (pid) is not the same as the ordering in the valid cubes list.'''
        if j is None:
            j = i.y
            k = i.z
            i = i.x
        return (i+self.map_rad)*2*(self.map_rad+1)+k+self.map_rad+1

    def ijk_to_rgb(self, i, j, k):
        c = Cube(i,j,k)
        return self.cube_to_rgb(c)

    def cube_to_rgb(self, c):
        if c in self.d_cube2rgb:
            return self.d_cube2rgb[c]
        else:
            if self.default:
                r, g, b = self.default
            else:
                i, j, k = c.x, c.y, c.z
                r, g, b = (int(i*self.scale+128), int(j*self.scale+128), int(k*self.scale+128))
            self.d_cube2rgb[c] = (r,g,b)
            return (r,g,b)
            
    def cube_to_terr(self, c):
        if c in self.d_cube2terr:
            return self.d_cube2terr[c]
        else:
            self.d_cube2terr[c] = Terrain.plains
            return self.d_cube2terr[c]
            
    def valid_pixel(self, x, y, r = 0):
        '''Determines whether a pixel is within the hexagon whose boundary is the center of all the hexes.
        (If the map doesn't use ocean outlying hexes, increase r accordingly.)'''
        return (round(-self.hex_size * (self.map_rad - 1) * self.s3) - r <= \
                x - self.mid_x <= \
                round(self.hex_size * (self.map_rad - 1) * self.s3) + r) and \
                (round(-self.hex_size*(self.map_rad - 1) * self.s3) - r <= \
                0.5 * ((x - self.mid_x) + self.s3 * (y - self.mid_y)) <= \
                round(self.hex_size * (self.map_rad - 1) * self.s3) + r) and \
                (round(-self.hex_size * (self.map_rad - 1) * self.s3) - r <= \
                0.5 * (self.s3 * (y - self.mid_y) - (x - self.mid_x)) <= \
                round(self.hex_size * (self.map_rad - 1) * self.s3) + r)
        
    @property
    def valid_cubes(self):
        if self.l_valid_cubes is not None:
            return self.l_valid_cubes
        valid = []
        for i in range(-self.map_rad,self.map_rad+1):
            for j in range(-self.map_rad,self.map_rad+1):
                for k in range(-self.map_rad,self.map_rad+1):
                    if i+j+k==0:
                        valid.append(Cube(i,j,k))
        self.l_valid_cubes = valid
        return valid

    def cubes2trio(self, trio):
        """Given a trio of cubes, return a trio of indices in valid_cubes."""
        if isinstance(trio[0], int):
            return trio
        for a, b in combinations(trio, 2):
            assert a.dist(b) == 1, f"Three cubes are not neighbors: {trio}."
        return tuple(sorted([self.valid_cubes.index(el) for el in trio]))
        
    def find_vertex(self, trio):
        """Finds the integer x,y pair that is the vertex between the three hexes.
        Can be provided either a truple of Cubes or a sorted tuple of indices."""
        trio = self.cubes2trio(trio) if isinstance(trio[0], Cube) else trio
        if trio in self.d_trio2vertex:
            return self.d_trio2vertex[trio]
        x = sum([self.cx[el] for el in trio])/3
        y = sum([self.cy[el] for el in trio])/3
        if self.crisp:
            self.d_trio2vertex[trio] = (int(x), int(y))
            return self.d_trio2vertex[trio]
        dists = [((x - self.cx[el])**2 + (y - self.cy[el])**2)/self.c2sigma[el] for el in trio]
        delta = max(dists) - min(dists)
        ldelta = 999999
        while delta < ldelta:
            lx = x
            ly = y
            ldelta = delta
            move_towards = np.argmax(dists)
            if abs(x - self.cx[trio[move_towards]]) > abs (y > self.cy[trio[move_towards]]):
                if x > self.cx[trio[move_towards]]:
                    x -= 1
                elif x < self.cx[trio[move_towards]]:
                    x += 1
            else:
                if y > self.cy[trio[move_towards]]:
                    y -= 1
                elif y < self.cy[trio[move_towards]]:
                    y += 1
            dists = [((x - self.cx[el])**2 + (y - self.cy[el])**2)/self.c2sigma[el] for el in trio]
            delta = max(dists) - min(dists)
        self.d_trio2vertex[trio] = (int(lx), int(ly))
        return (int(lx), int(ly))

    def edge_middle(self, hex_a, hex_b):
        """Find the x,y pair that are on the edge between hex_a and hex_b, equidistant to both centers (probably closest?)."""
        assert hex_a.dist(hex_b) == 1, "Cannot have edge between hexes that are not adjacent."
        vc_ids = tuple(sorted([self.valid_cubes.index(el) for el in [hex_a, hex_b]]))
        x, y = sum([self.cx[vc] for vc in vc_ids])/2, sum([self.cy[vc] for vc in vc_ids])/2
        if self.crisp:
            return (int(x), int(y))
        dists = [((x - self.cx[el])**2 + (y - self.cy[el])**2)/self.c2sigma[el] for el in vc_ids]
        delta = max(dists) - min(dists)
        ldelta = 999999
        while delta < ldelta:
            lx = x
            ly = y
            ldelta = delta
            move_towards = np.argmax(dists)
            if abs(x - self.cx[vc_ids[move_towards]]) > abs (y > self.cy[vc_ids[move_towards]]):
                if x > self.cx[vc_ids[move_towards]]:
                    x -= 1
                elif x < self.cx[vc_ids[move_towards]]:
                    x += 1
            else:
                if y > self.cy[vc_ids[move_towards]]:
                    y -= 1
                elif y < self.cy[vc_ids[move_towards]]:
                    y += 1
            dists = [((x - self.cx[el])**2 + (y - self.cy[el])**2)/self.c2sigma[el] for el in vc_ids]
            delta = max(dists) - min(dists)
    
    def calc_rivers(self, num_rivers, land_height):
        '''Calculate num_rivers rivers, adding them to self.d_trio2river'''
        for i in range(num_rivers):
            # Choose a random hex, weighted by height.
            source = random.choices(population=list(land_height.keys()), weights=list(land_height.values()), k=1)[0]
            worked = False
            while not worked:
                worked = self.source_river(source, land_height)

    def source_river(self, source, land_height) -> bool:
        '''Source is a cube. First find its highest vertex, and then extend it towards water. 
        Returns True if successful, False if restarting a previously started (or illegal) river.'''
        width = 0
        height = 0
        for rot in range(6):
            trio = [source, source.add(Cube(-1,0,1).rotate_right(rot)),source.add(Cube(-1,0,1).rotate_right(rot+1))]
            if trio[1] not in land_height or trio[2] not in land_height:
                continue
            tx, ty = self.find_vertex(trio)
            if self.topo[tx,ty] >= height:
                b3cubes = trio
                height = self.topo[tx,ty]
                sx, sy = tx, ty
        if height == 0:  # Couldn't find a vertex that doesn't touch water.
            return False
        start_trio = self.cubes2trio(b3cubes)
        if start_trio in self.d_trio2river:
            return False
        b3cubes = [self.valid_cubes[el] for el in start_trio]  # This reorders them, which is necessary for the next bit.
        b3cube_total = b3cubes[0].add(b3cubes[1]).add(b3cubes[2])
        # Find the lowest of the adjacent vertices.
        height = 256
        for away_cube_ind in range(3):
            # The next cube is the sum of the two cubes you follow minus the one you're moving away from.
            # For reasons, I'm calculating this as a+b+c-a-a instead of b+c-a.
            away_cube = b3cubes[away_cube_ind]
            next_cube = b3cube_total.sub(away_cube).sub(away_cube)
            next_trio = tuple(sorted([el for ind, el in enumerate(start_trio) if ind != away_cube_ind] + [self.valid_cubes.index(next_cube)]))
            tx, ty = self.find_vertex(next_trio)
            if self.topo[tx,ty] <= height:
                end_trio = next_trio
                end_away_cube = away_cube
                height = self.topo[tx,ty]
                end_xy = tx, ty
        assert height < 256
        # Now we have the end of the first river edge.
        self.d_trio2river[start_trio] = RiverEdge(width=width, start_trio=start_trio, start_xy=(sx,sy), end_trio=end_trio, end_xy=end_xy, source=True, merge=False)
        self.extend_river(end_trio, end_xy, end_away_cube, start_trio, width + 1, land_height)
        return True
            

    def extend_river(self, start_trio, start_xy, away_cube, source_trio, width, land_height):
        '''Given a river start_trio (the ending of the previous) and the cube it was moving away from, extend it towards the ocean.
        source_trio is used to delete 'source' from the first edge if this river merges into another.'''
        if start_trio in self.d_trio2river:
            self.d_trio2river[start_trio].source = False
            # TODO: Get this to not loop infinitely
            # self.join_flow(start_trio, width)
            return
        start_cubes = [self.valid_cubes[el] for el in start_trio]
        assert away_cube not in start_cubes
        # This cube is guaranteed to be included in the next trio.
        for t_ind, t_cube in enumerate(start_cubes):
            if t_cube.dist(away_cube) != 1:
                in_el = start_trio[t_ind]
        # Find which of the two directions is lower.
        height = 256
        for t_ind, t_cube in enumerate(start_cubes):
            if t_cube.dist(away_cube) != 1:
                continue
            next_cube = t_cube.add(t_cube).sub(away_cube)
            next_trio = tuple(sorted([start_trio[t_ind], in_el, self.valid_cubes.index(next_cube)]))
            tx, ty = self.find_vertex(next_trio)
            if self.topo[tx,ty] <= height:
                end_trio = next_trio
                height = self.topo[tx,ty]
                end_xy = (tx, ty)
        assert height < 256
        # Now we know which way to keep flowing.
        end3cubes = [self.valid_cubes[el] for el in end_trio]
        assert all([a.dist(b) == 1 for a,b in combinations(end3cubes, 2)])
        away_cube = [c for c in start_cubes if c not in end3cubes][0]
        if end_trio in self.d_trio2river:
            # We've run into a previous river. We need to figure out if it's a source or not.
            if self.d_trio2river[end_trio].source:
                self.d_trio2river[start_trio] = RiverEdge(width=width, start_trio=start_trio, start_xy=start_xy, end_trio=end_trio, end_xy=end_xy, source=False, merge=False)
            else:
                self.d_trio2river[start_trio] = RiverEdge(width=width, start_trio=start_trio, start_xy=start_xy, end_trio=end_trio, end_xy=end_xy, source=False, merge=True)
                self.d_trio2river[source_trio].source = False
        else:
            self.d_trio2river[start_trio] = RiverEdge(width=width, start_trio=start_trio, start_xy=start_xy, end_trio=end_trio, end_xy=end_xy, source=False, merge=False)
        if next_cube in land_height:  # If we haven't reached water, keep going.
            self.extend_river(end_trio, end_xy, away_cube, source_trio, width + 1, land_height)

    def join_flow(self, trio, width):
        '''Adds the width of a joining river to all downstream edges.'''
        cycle_prevention = set()
        while trio in self.d_trio2river and trio not in cycle_prevention:
            self.d_trio2river[trio].width += width
            cycle_prevention.add(trio)
            trio = self.d_trio2river[trio].end_trio
    
    def provinces(self, filedir=None):
        '''Create provinces.bmp. Uses d_cube2rgb, choosing colors based on position if any are empty.Saves if filedir is passed.'''
        pixels = self.img_provinces.load()
        last_i, last_j, last_k = (-999, -999, -999)
        last_r, last_g, last_b = IMPASSABLE
        for x in range(self.max_x):
            for y in range(self.max_y):
                #check if inside big hex
                if self.valid_pixel(x,y):
                    i, j, k = self.pixel_to_cube(x,y).tuple()
                    if not( i==last_i and j==last_j and k==last_k ):
                        last_r, last_g, last_b = self.ijk_to_rgb(i,j,k)
                        last_i, last_j, last_k = i, j, k
                    pixels[x,y] = (last_r, last_g, last_b)
                else:
                    pixels[x,y] = IMPASSABLE
        if filedir:
            self.save_provinces(filedir)

    def save_provinces(self, filedir):
        self.img_provinces.save(os.path.join(filedir, 'provinces.png'))

    def heightmap(self, land_height, water_depth, waste_list, filedir=''):
        '''Create heightmap.png. 16 is the boundary?'''
        self.topo = np.zeros((self.max_x,self.max_y))
        last_cube = None
        last_range = (0,0)
        for x in range(self.max_x):
            for y in range(self.max_y):
                if self.valid_pixel(x,y):
                    c = self.pixel_to_cube(x,y)
                    if c != last_cube:
                        if c in land_height:
                            last_height = 96 + land_height[c] * 3
                        elif c in water_depth:
                            last_height = 94 - water_depth[c] * 10
                        else:
                            last_height = 0
                        if c in self.d_cube2terr:
                            last_range = TERRAIN_HEIGHT[self.d_cube2terr[c]]
                        else:
                            last_range = (0,0)
                        last_cube = c
                    self.topo[x,y] = last_height + random.randint(last_range[0], last_range[1])
                else:
                    self.topo[x,y] = 0
        # TODO: Gaussian filtering
        self.img_topology.putdata(self.topo.transpose().flatten())
        if filedir:
            with open(os.path.join(filedir, 'heightmap.heightmap'), 'w') as f:
                f.write("heightmap_file=\"map_data/packed_heightmap.png\"\n")
                f.write("indirection_file=\"map_data/indirection_heightmap.png\"\n")
                f.write(f"original_heightmap_size={{ {self.max_x} {self.max_y} }}\n")
                f.write("tile_size=33\n")
                f.write("should_wrap_x=no\n")
                f.write("level_offsets={ { 0 0 }{ 0 0 }{ 0 0 }{ 0 0 }{ 0 7 }}\n")
            self.img_topology.save(os.path.join(filedir, 'heightmap.png'))
            
    def rivers(self, land_height, bname_from_pid, num_rivers, last_pid, filedir=''):
        '''Create rivers.png.
        This will possibly edit provinces.png, definition.csv, and adjacencies.csv if it creates any major rivers.'''
        self.calc_rivers(num_rivers, land_height)
        r_pixels = self.img_rivers.load()
        lc = Cube(0,0,0)
        def_addenda = []
        adj_addenda = []
        # Determine some palette colors.
        # Figure out all the possible river widths:
        widths = sorted({riv.width for riv in self.d_trio2river.values()})
        if widths[-1] > MAJOR_RIVER_THRESHOLD:
            p_pixels = self.img_provinces.load()
            print("At least one major river will be made.\n")
        width_mapping = {width: 3 + i if width < MAJOR_RIVER_THRESHOLD else 254 for i, width in enumerate(widths)}
        # Set the background colors.
        for x in range(self.max_x):
            for y in range(self.max_y):
                #check if inside big hex
                if self.valid_pixel(x, y):
                    c = self.pixel_to_cube(x, y)
                    if c != lc:
                        if c in land_height:
                            lpix = LAND
                        else:
                            lpix = WATER
                        lc = c
                    r_pixels[x,y] = lpix
                else:
                    r_pixels[x,y] = WATER
        # Now we can start making rivers; each edge can be done independently.
        for start_trio, riv in self.d_trio2river.items():
            if riv.width < MAJOR_RIVER_THRESHOLD:
                self.paint_edge(riv.start_xy, riv.end_xy, width_mapping[riv.width], r_pixels, riv.merge, extend=not(riv.end_trio in self.d_trio2river) or self.d_trio2river[riv.end_trio].width >= MAJOR_RIVER_THRESHOLD)
                if riv.source:
                    r_pixels[riv.start_xy[0], riv.start_xy[1]] = SOURCE
            else:
                da, aa = self.paint_major_river(riv, r_pixels, p_pixels, last_pid, bname_from_pid)
                def_addenda.append(da)
                adj_addenda.append(aa)
                last_pid += 1
        if len(def_addenda) > 0:
            with open(os.path.join(filedir, 'map_data', 'definition.csv'), 'a') as f:
                f.write("".join(def_addenda))
            with open(os.path.join(filedir, 'map_data', 'adjencies.csv'), 'a') as f:
                f.write("".join(adj_addenda))
            self.save_provinces(filedir)
        self.img_rivers.save(os.path.join(filedir, 'map_data', 'rivers.png'))
        return last_pid

    def paint_edge_crisp(self, start_xy, end_xy, paint, pixels, merge: bool = False, extend=False):
        '''Paint a river segment from start_xy to end_xy. Crisp maps only!
        If extend is true, will continue in the same direction for RIVER_EXTEND pixels.'''
        assert not(merge and extend)  # You should never want to do both.
        print(start_xy, end_xy)
        if start_xy[1] == end_xy[1]:  # Only horizontal edges are possible.
            extend = RIVER_EXTEND if extend else 0
            # We don't do the last pixel because it would maybe overwrite the next one.
            # Does this actually matter? idk, it should only matter for source, and we'll never do that, maybe.
            # It is important for getting merges right tho.
            direction = 1 if end_xy[0] > start_xy[0] else -1
            dest = end_xy[0] - direction + extend * direction
            for x in range(start_xy[0], dest, direction):
                pixels[x, start_xy[1]] = paint
            if merge:
                pixels[dest, start_xy[1]] = MERGE
            else:
                pixels[dest, start_xy[1]] = paint
        else: # We have an angled edge.
            del_x = (1,0) if end_xy[0] > start_xy[0] else (-1,0)
            del_y = (0,1) if end_xy[1] > start_xy[1] else (0,-1)
            x_dist = abs(end_xy[0] - start_xy[0])
            x_path = [del_x] * x_dist
            y_dist = abs(end_xy[1] - start_xy[1])
            y_path = [del_y] * y_dist
            if extend:
                p = x_dist / (x_dist + y_dist)
                for _ in range(RIVER_EXTEND):
                    if random.random() < p:  # big x => high p => add more xs. 
                        x_path.append(del_x)
                        x_dist += 1
                    else:
                        y_path.append(del_y)
                        y_dist += 1
            m_dist = x_dist + y_dist
            # Interleaving the two paths; stolen from Stack Overflow.
            path = [x[1] for x in heapq.merge(zip(count(0, y_dist), x_path), zip(count(0, x_dist), y_path))]
            cx, cy = start_xy
            for x,y in path[:-1]:
                pixels[cx, cy] = paint
                cx += x
                cy += y
            if merge:
                pixels[cx, cy] = MERGE
            else:
                pixels[cx, cy] = paint

    def paint_edge_non_crisp(self, start_xy, end_xy, paint, pixels):
        '''Paint a river segment from start_xy to end_xy.'''
        raise NotImplementedError

    def paint_major_river_crisp(self, riv, r_pixels, p_pixels, pid, bname_from_pid):
        '''Paint a major river segment defined by riv. Crisp maps only!'''
        rgb = new_rgb(self.d_cube2rgb)
        self.d_cube2rgb[riv.start_trio] = rgb
        if riv.start_xy[1] == riv.end_xy[1]:  # Only horizontal edges are possible.
            for x in range(riv.start_xy[0], riv.end_xy[0]):
                r_pixels[x, riv.start_xy[1] - 1] = WATER
                r_pixels[x, riv.start_xy[1]] = WATER
                r_pixels[x, riv.start_xy[1] + 1] = WATER
                p_pixels[x, riv.start_xy[1] - 1] = rgb
                p_pixels[x, riv.start_xy[1]] = rgb
                p_pixels[x, riv.start_xy[1] + 1] = rgb
        else: # We have an angled edge.
            del_x = (1,0) if riv.end_xy[0] - riv.start_xy[0] > 0 else (-1,0)
            del_y = (0,1) if riv.end_xy[1] - riv.start_xy[1] > 0 else (0,-1)
            x_dist = abs(riv.end_xy[0] - riv.start_xy[0])
            x_path = [del_x] * x_dist
            y_dist = abs(riv.end_xy[1] - riv.start_xy[1])
            y_path = [del_y] * y_dist
            m_dist =  x_path + y_path
            # Interleaving the two paths; stolen from Stack Overflow.
            path = [x[1] for x in heapq.merge(zip(count(0, y_dist), x_path), zip(count(0, x_dist), y_path))]
            cx, cy = riv.start_xy
            for x,y in path:
                r_pixels[cx, cy] = WATER
                r_pixels[cx + del_x[0], cy] = WATER
                r_pixels[cx, cy + del_y[1]] = WATER
                p_pixels[cx, cy] = rgb
                p_pixels[cx + del_x[0], cy] = rgb
                p_pixels[cx, cy + del_y[1]] = rgb
                cx += x
                cy += y
        def_addendum = ";".join([str(s) for s in [pid, *rgb, riv.name, "x", "\n"]])
        midx = (riv.end_xy[0] + riv.start_xy[0]) // 2
        midy = (riv.end_xy[1] + riv.start_xy[1]) // 2
        shore1 = (midx + del_x[0] * 2, midy - del_y[1] * 2)
        shore2 = (midx - del_x[0] * 2, midy + del_y[1] * 2)
        pid1 = self.d_cube2pid[self.pixel_to_cube(*shore1)]
        pid2 = self.d_cube2pid[self.pixel_to_cube(*shore2)]
        comment = f"{pid1}-{pid2}\n"
        # From;To;Type;Through;start_x;start_y;stop_x;stop_y;Comment
        adj_addendum = ";".join([str(s) for s in [pid1, pid2, "sea", pid, shore1[0], shore1[1], shore2[0], shore2[1], comment]])
        return def_addendum, adj_addendum

    def paint_major_river_non_crisp(self, riv, r_pixels, p_pixels, pid, bname_from_pid):
        '''Paint a major river segment defined by riv.'''
        raise NotImplementedError
