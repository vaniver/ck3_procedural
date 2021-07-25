import PIL.Image
from cube import Cube
from math import sqrt
import numpy as np
import os
import pickle
import random

def dist2line(x, y, srcx, srcy, destx, desty):
    return abs((desty-srcy)*x-(destx-srcx)*y+destx*srcy-desty*srcx)/sqrt((desty-srcy)**2+(destx-srcx)**2)

class CK2Map:
    
    def __init__(self, max_x=8192, max_y=4096, hex_size=20, map_size=40, crisp=True, default=None):
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
        self.img_world_normal_height = PIL.Image.new('RGB', (max_x,max_y),  "white")
        self.img_rivers = PIL.Image.new('P', (max_x,max_y),  255)  # 255=white
        with open(os.path.join("data", "rivers_palette.txt"), "r") as f:
            self.img_rivers.putpalette([int(s) for s in f.read().split(',')])
        self.d_cube2rgb = {}
        self.d_cube2terr = {}
        self.d_cube2pid = {}
        self.d_pid2cube = {}
        self.trio2vertex = {}
        self.river_width = {}
        self.river2river = {}
        self.river_merges = []
        self.river_sources = []
        self.s3 = sqrt(3)
        self.l_valid_cubes = None
        self.crisp = crisp
        if not self.crisp:
            self.cx = np.zeros(self.num_hexes)
            self.cy = np.zeros(self.num_hexes)
            self.sigma = np.zeros(self.num_hexes)
            self.c2cx = {}
            self.c2cy = {}
            self.c2sigma = {}
            for ind, c in enumerate(self.valid_cubes()):
                i = c.x
                j = c.y
                k = c.z
                self.cx[ind] = int(i * 1.5 * self.hex_size + self.mid_x)
                self.cy[ind] = int((j * 3 * self.hex_size + self.cx[ind] - self.mid_x) / self.s3 + self.mid_y)
                self.sigma[ind] = np.random.rand() + 0.5
                self.c2cx[c] = self.cx[ind]
                self.c2cy[c] = self.cy[ind]
                self.c2sigma[c] = self.sigma[ind]
                
    def pixel_to_cube(self, x,y):
        if self.crisp:
            q = (x-self.mid_x) * 2.0/3.0/self.hex_size
            r = (-(x-self.mid_x) / 3.0 + self.s3/3.0 * (y-self.mid_y))/self.hex_size
            return self.cube_round(q,-q-r,r)
        return self.valid_cubes()[(((self.cx-x)**2+(self.cy-y)**2)/self.sigma).argmin()]
        
    def cube_round(self, i, j, k):
        ri, rj, rk = int(round(i)), int(round(j)), int(round(k))
        di, dj, dk = abs(ri-i), abs(rj-j), abs(rk-k)

        if di > dj and di > dk:
            ri = -rj-rk
        elif dj > dk:
            rj = -ri-rk
        else:
            rk = -ri-rj
        return (ri, rj, rk)

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
    
    def cube_to_vert(self, c):
        if c in self.d_cube2vert:
            return self.d_cube2vert[c]
        else:
            i, j, k = c.x, c.y, c.z
            rad = max(abs(i), abs(j), abs(k))
            self.d_cube2vert[c] = round(140 - 100 * rad / self.map_rad)
            return self.d_cube2vert[c]
            
    def cube_to_terr(self, c):
        if c in self.d_cube2terr:
            return self.d_cube2terr[c]
        else:
            self.d_cube2terr[c] = 15
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
                
    def xy2height(self, x, y, land_height, water_height, waste_list):
        q = (x-self.mid_x) * 2.0/3.0/self.hex_size
        r = (-(x-self.mid_x) / 3.0 + self.s3/3.0 * (y-self.mid_y))/self.hex_size
        ce = Cube(*self.cube_round(q,r,-q-r)) #The crisp center, which is always used for topology to determine the relevant triangle.
        if ce.mag() >= self.map_rad:
            return 17
        t = []
        if True:
            combo = sorted([ce] + list(ce.neighbors()), key = str)
            dists = [((self.c2cx[el]-x)**2+(self.c2cy[el]-y)**2) for el in combo]
            sd_in = sorted(range(len(dists)), key=lambda k: dists[k])
            n0, n1, n2 = [combo[el] for el in sd_in[:3]] #For some reason the center isn't always the center???
        else:
            dists = ((self.cx-x)**2+(self.cy-y)**2)
            sd_in = sorted(range(len(dists)), key=lambda k: dists[k])
            n0, n1, n2 = [self.valid_cubes()[el] for el in sd_in[:3]]
        
        if int(self.c2cx[n0])==x and int(self.c2cy[n0])==y: #This both saves times sometimes, and rescues us from having a non-triangular area, 
            #which can happen if on the center and all distances are equal.
            share = [1,0,0]
        else:
            share = [0,0,0]
            share[0] = dist2line(x,y,self.c2cx[n1],self.c2cy[n1],self.c2cx[n2],self.c2cy[n2])
            share[1] = dist2line(x,y,self.c2cx[n0],self.c2cy[n0],self.c2cx[n2],self.c2cy[n2])
            share[2] = dist2line(x,y,self.c2cx[n0],self.c2cy[n0],self.c2cx[n1],self.c2cy[n1])
        center = self.valid_cubes()[(((self.cx-x)**2+(self.cy-y)**2)/self.sigma).argmin()] #The true center, used to check land/water.
        ss = sum(share)
        if ss > 0:
            share = [el/ss for el in share]
        else:
            share = [1, 0, 0]
        if center == n1:
            temp = share[0]
            share[0] = share[1]
            share[1] = temp
        if center == n2:
            temp = share[0]
            share[0] = share[2]
            share[2] = temp
        if center in land_height:
            h = 18 + land_height[center] * 3 * share[0]
            for el in [n0, n1, n2]:
                if el in self.d_cube2terr:
                    t.append(self.d_cube2terr[el])
                else:
                    t.append(-1)
            if t[0] == 15:
                return 16 #Lakes have to have water topology.
            if n1 in land_height:
                h += land_height[n1] * 3 * share[1]
            if n2 in land_height:
                h += land_height[n2] * 3 * share[2]
            if t[0] == 9:
                h += 80 * max(0, share[0]-0.5)
                if center in waste_list:
                    h += 80 * max(0, share[0]-0.4)
                if t[1] == 9:
                    h += 30 * share[1]
                    if n1 in waste_list:
                        h += 30 * share[1]
                if t[2] == 9:
                    h += 30 * share[2]
                    if n2 in waste_list:
                        h += 30 * share[2]
            elif t[0] == 8:
                h += 20 * max(0, share[0]-0.4)
                if t[1] == 9:
                    h += 15 * share[1]
                    if n1 in waste_list:
                        h += 15 * share[1]
                if t[2] == 9:
                    h += 15 * share[2]
                    if n2 in waste_list:
                        h += 15 * share[2]
            return max(96,min(int(h),240))
        else:
            w_h = [94, 89, 78, 62, 41, 17]#[94-round(5*el**1.7) for el in range(6)]
            h = [17, 17, 17]
            if center in water_height:
                h[0] = w_h[min(water_height[center],5)]
            if n1 in water_height:
                h[1] = w_h[min(water_height[n1],5)]
            elif n1 in land_height:
                h[1] = 94
            if n2 in water_height:
                h[2] = w_h[min(water_height[n2],5)]
            elif n2 in land_height:
                h[2] = 94
            return max(17,min(int(sum([h[ind]*share[ind] for ind in range(3)])),94))
                
    def cubes2trio(self, trio):
        """Given a trio of cubes, return a trio of indices in valid_cubes."""
        for a, b in combinations(trio, 2):
            assert a.dist(b) == 1, f"Three cubes are not neighbors: {trio}."
        return tuple(sorted([self.valid_cubes().index(el) for el in trio]))
        
    def find_boundary(self, trio):
        """Finds the integer x,y pair that is the vertex between the three hexes."""
        t = self.cubes2trio(trio)
        if t in self.trio2vertex:
            return self.trio2vertex[t]
        x = sum([self.c2cx[el] for el in trio])/3
        y = sum([self.c2cy[el] for el in trio])/3
        if self.crisp:
            self.trio2vertex[t] = (int(x), int(y))
            return self.trio2vertex[t]
        dists = [((x - self.c2cx[el])**2 + (y - self.c2cy[el])**2)/self.c2sigma[el] for el in trio]
        delta = max(dists) - min(dists)
        ldelta = 999999
        while delta < ldelta:
            lx = x
            ly = y
            ldelta = delta
            move_towards = np.argmax(dists)
            if abs(x - self.c2cx[trio[move_towards]]) > abs (y > self.c2cy[trio[move_towards]]):
                if x > self.c2cx[trio[move_towards]]:
                    x -= 1
                elif x < self.c2cx[trio[move_towards]]:
                    x += 1
            else:
                if y > self.c2cy[trio[move_towards]]:
                    y -= 1
                elif y < self.c2cy[trio[move_towards]]:
                    y += 1
            dists = [((x - self.c2cx[el])**2 + (y - self.c2cy[el])**2)/self.c2sigma[el] for el in trio]
            delta = max(dists) - min(dists)
        self.trio2vertex[t] = (int(lx), int(ly))
        return (int(lx), int(ly))

    def edge_middle(self, hex_a, hex_b):
        """Find the x,y pair that are on the edge between hex_a and hex_b, equidistant to both centers (probably closest?)."""
        assert hex_a.dist(hex_b) == 1, "Cannot have edge between hexes that are not adjacent."
        vc_ids = tuple(sorted([self.valid_cubes().index(el) for el in hex_a, hex_b]))
        x = sum([self.c2cx[el] for el in vc_ids])/2
        y = sum([self.c2cy[el] for el in vc_ids])/2
        if self.crisp:
            return (int(x), int(y))
        dists = [((x - self.c2cx[el])**2 + (y - self.c2cy[el])**2)/self.c2sigma[el] for el in vc_ids]
        delta = max(dists) - min(dists)
        ldelta = 999999
        while delta < ldelta:
            lx = x
            ly = y
            ldelta = delta
            move_towards = np.argmax(dists)
            if abs(x - self.c2cx[vc_ids[move_towards]]) > abs (y > self.c2cy[vc_ids[move_towards]]):
                if x > self.c2cx[vc_ids[move_towards]]:
                    x -= 1
                elif x < self.c2cx[vc_ids[move_towards]]:
                    x += 1
            else:
                if y > self.c2cy[vc_ids[move_towards]]:
                    y -= 1
                elif y < self.c2cy[vc_ids[move_towards]]:
                    y += 1
            dists = [((x - self.c2cx[el])**2 + (y - self.c2cy[el])**2)/self.c2sigma[el] for el in vc_ids]
            delta = max(dists) - min(dists)
    
    def old_make_river(self, source, land_height, water_list, rot=None):
        '''Source is a cube. Find the highest vertex, or the one defined by -1,0,1 rotated right by rot, and then flow to the sea.'''
        height = 0
        if rot:
            btrio = [source, source.add(Cube(-1,0,1).rotate_right(rot)),source.add(Cube(-1,0,1).rotate_right(rot+1))]
            bx, by = self.find_boundary(btrio)
        else:
            num_found = 0
            for rot in range(6):
                trio = [source, source.add(Cube(-1,0,1).rotate_right(rot)),source.add(Cube(-1,0,1).rotate_right(rot+1))]
                tx, ty = self.find_boundary(trio)
                if self.topo[tx,ty] > height:
                    bx, by = tx, ty
                    btrio = trio
                    num_found = 1
                #elif self.topo[nx,ny] == low_topo:
                #    num_found += 1
                #    if random.random() < 1.0/num_found:
                #        bx, by = nx, ny
        width = 0
        prev = self.cubes2trio(btrio)
        valid = prev not in self.river2river
        if valid:
            self.river_sources.append(self.cubes2trio(btrio))
        while valid:
            trio = [self.valid_cubes()[el] for el in prev]
            prev_lh = sum([land_height[el] for el in trio if el in land_height])
            #Figure out where we could go.
            opt = []
            for a in range(3):
                b = (a + 1) % 3
                c = 3-a-b #0, 1 -> 2, 1,2 -> 0, 2,0 -> 1
                new_trio = [trio[a], trio[b], trio[a].add(trio[b].sub(trio[c]))]
                if trio[a] not in water_list and trio[b] not in water_list and not (self.cubes2trio(new_trio) in self.river2river and
                                                                                    self.river2river[self.cubes2trio(new_trio)] == prev):
                    if sum([land_height[el] for el in new_trio if el in land_height]) < prev_lh:
                        opt.append(new_trio)
            if len(opt) > 0:
                random.shuffle(opt)
                ntrio = self.cubes2trio(opt[0])
                self.river2river[prev] = ntrio
                self.river_width[(prev, ntrio)] = width
                prev = ntrio
                if ntrio in self.river2river: #We're joining a pre-existing river.
                    valid = False
                    if ntrio in self.river_sources:
                        self.river_sources.remove(ntrio)
                    else:
                        self.river_merges.append(ntrio)
                        if width <= self.river_width[(ntrio, self.river2river[ntrio])]:
                            self.river_sources.remove(self.cubes2trio(btrio))
                        else:
                            #Trace back and remove the other river's source instead.
                            for psrc in self.river_sources:
                                if psrc != self.cubes2trio(btrio):
                                    next_loc = psrc
                                    while next_loc in self.river2river:
                                        next_loc = self.river2river[next_loc]
                                        if next_loc == ntrio:
                                            self.river_sources.remove(psrc) 
                    while ntrio in self.river2river:
                        self.river_width[(ntrio,self.river2river[ntrio])] += max(width,1)
                        ntrio = self.river2river[ntrio]
                width += 1
                if opt[0][-1] in water_list: #We reached a body of water.
                    valid = False
            else:
                if prev in self.river_sources:
                    self.river_sources.remove(prev)
                valid = False
    
    def provinces(self, filedir=''):
        '''Create provinces.bmp. Uses d_cube2rgb, choosing colors based on position if any are empty.'''
        pixels = self.img_provinces.load()
        last_i, last_j, last_k = (-999, -999, -999)
        last_r, last_g, last_b = (0, 0, 0)
        for x in range(self.max_x):
            for y in range(self.max_y):
                #check if inside big hex
                if self.valid_pixel(x,y):
                    if self.crisp: #This is a giant hack, should be fixable
                        i, j, k = self.pixel_to_cube(x,y)#.tuple()
                    else:
                        i, j, k = self.pixel_to_cube(x,y).tuple()
                    if not( i==last_i and j==last_j and k==last_k ):
                        last_r, last_g, last_b = self.ijk_to_rgb(i,j,k)
                        last_i, last_j, last_k = i, j, k
                    pixels[x,y] = (last_r, last_g, last_b)
                else:
                    pixels[x,y] = (0, 0, 0)
        if filedir:
            self.img_provinces.save(os.path.join(filedir, 'provinces.png'))

    def heightmap(self, land_height, water_height, waste_list, terrain_height, filedir=''):
        '''Create heightmap.png. 16 is the boundary?'''
        self.topo = np.zeros((self.max_x,self.max_y))
        last_cube = None
        for x in range(self.max_x):
            for y in range(self.max_y):
                if self.valid_pixel(x,y):
                    c = self.pixel_to_cube(x,y)
                    if c != last_cube:
                        if c in land_height:
                            dist_height = land_height[c] * 30
                        elif c in water_height:
                            dist_height = water_height[c] * 30
                        else:
                            dist_height = 0
                        # last_cube = c
                        # if c in self.d_cube2terr:
                        #     dist_height += terrain_height[self.d_cube2terr[c]]
                        # else:
                        #     dist_height += 16
                        last_height = dist_height  # max(0, dist_height)
                    self.topo[y,x] = last_height
                else:
                    self.topo[y,x] = 0
        self.img_topology.putdata(self.topo.flatten())
        if filedir:
            with open(os.path.join(filedir, 'heightmap.heightmap'), 'w') as f:
                f.write("heightmap_file=\"map_data/packed_heightmap.png\"\n")
                f.write("indirection_file=\"map_data/indirection_heightmap.png\"\n")
                f.write("original_heightmap_size={ 8192 4096 }\n")
                f.write("tile_size=33\n")
                f.write("should_wrap_x=no\n")
                f.write("level_offsets={ { 0 0 }{ 0 0 }{ 0 0 }{ 0 0 }{ 0 7 }}\n")
            self.img_topology.save(os.path.join(filedir, 'heightmap.png'))
            
    def old_topology(self, land_height, water_height, waste_list, filedir=''):
        '''Create topology.bmp. 95 is the boundary. Also creates world_normal_height.'''
        self.topo = np.zeros((self.max_x,self.max_y))
        for x in range(self.max_x):
            for y in range(self.max_y):
                self.topo[x,y] = self.xy2height(x,y,land_height,water_height,waste_list)
        self.img_topology.putdata(self.topo.transpose().flatten())
        if filedir:
            self.img_topology.save(os.path.join(filedir, 'topology.bmp'))
        pixels = self.img_world_normal_height.load()
        for x in range(self.max_x):
            for y in range(self.max_y):
                xComp = min(11,max(-11, self.topo[max(0,x-1),y]-self.topo[min(self.max_x-1, x+1),y]))
                yComp = min(11,max(-11, self.topo[x,max(0,y-1)]-self.topo[x,min(self.max_y-1, y+1)]))
                xComp *= abs(xComp)
                yComp *= abs(yComp)
                zComp = sqrt(max(0,127*127 - xComp*xComp - yComp*yComp))
                pixels[x,y] = (int(xComp + 128), int(yComp + 128), int(zComp + 128))
        if filedir:
            self.img_world_normal_height.save(os.path.join(filedir, 'world_normal_height.bmp'))
            
    def terrain(self, filedir='', edge=15):
        '''Create terrain.bmp. Uses d_cube2terr, defaulting to water.
        Terrain is the same for the whole hex, which sort of sucks.'''
        pixels = self.img_terrain.load()
        lc = Cube(0,0,0)
        for x in range(self.max_x):
            for y in range(self.max_y):
                    #check if inside big hex
                    if self.valid_pixel(x,y):
                        c = self.pixel_to_cube(x,y)
                        if c != lc:
                            if c in self.d_cube2terr:
                                lterr = self.d_cube2terr[c]
                            else:
                                lterr = 15
                            lc = c
                        #Check for water or 'beach'
                        if any([self.topo[x,y] < 95, self.topo[max(0,x-1),y] < 95, self.topo[min(self.max_x-1, x+1),y] < 95,
                                self.topo[x,max(0,y-1)] < 95, self.topo[x,min(self.max_y-1, y+1)] < 95]):
                            pixels[x,y] = 15
                        else:
                            if lterr == 9: #Mountain, we need to check the height
                                if self.topo[x,y] > 170:
                                    pixels[x,y] = 11
                                elif self.topo[x,y] > 150:
                                    pixels[x,y] = 10
                                else:
                                    pixels[x,y] = lterr
                            elif lterr == 16:
                                pixels[x,y] = 8
                            else: #Normal
                                pixels[x,y] = lterr
                    else:
                        pixels[x,y] = 15
        if filedir:
            self.img_terrain.save(os.path.join(filedir, 'terrain.bmp'))
            
    def rivers(self, land_height, filedir='', logfile=''):
        '''Create rivers.bmp.'''
        pixels = self.img_rivers.load()
        if logfile:
            logf = open(logfile)
        lc = Cube(0,0,0)
        rw = list(range(3, 7)) + list(range(8, 12))
        for x in range(self.max_x):
            for y in range(self.max_y):
                    #check if inside big hex
                    if self.valid_pixel(x, y):
                        c = self.pixel_to_cube(x, y)
                        if c != lc:
                            if c in land_height:
                                lpix = 255
                            else:
                                lpix = 254
                            lc = c
                        pixels[x,y] = lpix
                    else:
                        pixels[x,y] = 254
        for true_width in range(max(self.river_width.values())+1):
            for src, dest in self.river2river.items():
                if self.river_width[src, dest] == true_width:
                    strio = [self.valid_cubes()[el] for el in src]
                    dtrio = [self.valid_cubes()[el] for el in dest]
                    srcx, srcy = self.find_boundary(strio)
                    destx, desty = self.find_boundary(dtrio)
                    width = min(self.river_width[src,dest],7) #Rivers can only be so wide.
                    x, y =  srcx, srcy
                    path = [False]*abs(destx-srcx)+[True]*abs(desty-srcy)
                    random.shuffle(path)
                    if destx > srcx:
                        xdir = 1
                    if destx < srcx:
                        xdir = -1
                    if desty > srcy:
                        ydir = 1
                    if desty < srcy:
                        ydir = -1
                    valid = True
                    for index in range(len(path)):
                        turn = path[index]
                        if turn:
                            y+=ydir
                            nnbors = sum([pixels[a]<13 for a in [(xx,y) for xx in [x-1, x+1]]])
                            ahead = pixels[(x,y+ydir)]<13
                            nanbors = sum([pixels[a]<13 for a in [(xx,y+ydir) for xx in [x-1, x+1]]])
                        else:
                            x+=xdir
                            nnbors = sum([pixels[a]<13 for a in [(x,yy) for yy in [y-1, y+1]]])
                            ahead = pixels[(x+xdir,y)]<13
                            nanbors = sum([pixels[a]<13 for a in [(x+xdir,yy) for yy in [y-1, y+1]]])
                        if nnbors == 0 and valid:
                            pixels[x,y] = rw[width]
                            if ahead:
                                valid = False
                                #Check to make sure we're not hitting a spur.
                                if nanbors > 0:
                                    if logfile:
                                        logf.write("Check for a spur at {},{}".format(x,y))
                                    else:
                                        print("Check for a spur at {},{}".format(x,y))
                        elif not turn in path[index:] and valid:
                            if turn:
                                x+=xdir
                                y-=ydir
                                nnbors = sum([pixels[a]<13 for a in [(x,yy) for yy in [y-1, y+1]]])
                            else:
                                x-=xdir
                                y+=ydir
                                nnbors = sum([pixels[a]<13 for a in [(xx,y) for xx in [x-1, x+1]]])
                            path[index+path[index:].index(not turn)] = turn
                            path[index] = not turn
                            if nnbors == 0 and valid:
                                pixels[x,y] = rw[width]
                            else:
                                valid = False
                        elif not ahead and valid:
                            pixels[x,y] = rw[width]
                            valid = False
                        else:
                            valid = False
                    if dest not in self.river2river: #We've terminated, so go a bit further.
                        for turn in path[-2:]:
                            if turn:
                                y+=ydir
                            else:
                                x+=xdir
                            pixels[x,y] = rw[width]
        for src in self.river_sources:
            strio = [self.valid_cubes()[el] for el in src]
            srcx, srcy = self.find_boundary(strio)
            pixels[srcx, srcy] = 0
        for merge in self.river_merges:
            loc = [self.valid_cubes()[el] for el in merge]
            x, y = self.find_boundary(loc)
            xys = [(xx,y) for xx in [x-1, x+1]] + [(x,yy) for yy in [y-1, y+1]]
            pix = [pixels[a] for a in [(xx,y) for xx in [x-1, x+1]] + [(x,yy) for yy in [y-1, y+1]]]
            pixord = np.argsort(pix)
            if pix[pixord[0]]<pix[pixord[1]]:
                pixels[xys[pixord[0]]] = 1
            else:
                pixels[xys[pixord[0]]] = 1
                if logfile:
                    logf.write("I guessed which river pixel to merge, at around {}. Please fix.".format((x,y)))
                else:
                    print("I guessed which river pixel to merge, at around {}. Please fix.".format((x,y)))
        if filedir:
            self.img_rivers.save(os.path.join(filedir, 'rivers.png'))
        if logfile:
            logf.close()