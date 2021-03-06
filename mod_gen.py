from dataclasses import dataclass
from enum import Enum
import math
import os
import pickle
import random
import shutil
from typing import List

import numpy as np

from cube import Cube
from tile import Tile
from ck2map import CK2Map
import continent_gen


Terrain = Enum('Terrain','plains farmlands hills mountains desert desert_mountains oasis jungle forest taiga wetlands steppe floodplains drylands')
NUM_KINGDOM_HEXES = sum([sum(x) for x in continent_gen.KINGDOM_SIZE_LIST])
NUM_CENTER_HEXES = sum(continent_gen.CENTER_SIZE_LIST)
NUM_BORDER_HEXES = sum(continent_gen.BORDER_SIZE_LIST)

def make_dot_mod(file_dir, mod_name, mod_disp_name):
    '''Build the basic mod details file.'''
    shared = "version = 0.0.1\n"
    shared += "tags = {\n\t\"Total Conversion\"\n}\n"
    shared += "name = \"{}\"\n".format(mod_disp_name)
    shared += "supported_version = 1.2.2\n"
    outer = "path = \"mod/{}\"\n".format(mod_name)
    
    # replace_paths = ["common/landed_titles", "map_data"] #"common/bookmarks", "common/cultures", "common/dynasties", 
    #                     #"common/offmap_powers", "history/characters", "history/offmap_powers", "history/provinces",
    #                     #"history/technology", "history/titles", "history/wars"]
    # outer += "replace_path = \"" + "\"\nreplace_path = \"".join(replace_paths)+"\""
    os.makedirs(os.path.join(file_dir, mod_name), exist_ok=True)
    with open(os.path.join(file_dir,"{}.mod".format(mod_name)),'w') as f:
        f.write(shared + outer)
    with open(os.path.join(file_dir, mod_name, "descriptor.mod".format(mod_name)),'w') as f:
        f.write(shared)


@dataclass
class Kingdom:
    name: str = ""
    island: bool = False
    religion: str = ""


@dataclass
class Empire:
    name: str = ""
    angle: int = 0
    kingdoms: List["Kingdom"] = []
    base_terrain: Terrain = Terrain.plains
    waste_terrain: Terrain = Terrain.plains
    continentals: int = 0
    islands: int = 0

    def add_kingdom(self, kingdom):
        self.kingdoms.append(kingdom)
        if kingdom.island:
            self.islands += 1
        else:
            self.continentals += 1

    def continental_names(self):
        return [k for k in self.kingdoms if not k.island]

    def island_names(self):
        return [k for k in self.kingdoms if k.island]

def read_config_file(config_filepath):
    '''Read the config file for the map.
    This ignores everything after a #, and expects rows to look like:
    empire_name kingdom_name    religion    continent/island
    separated by tabs.'''
    config_file = open(config_filepath)
    empires = {}
    base_terrain_from_empire = {}
    waste_terrain_from_empire = {}
    kingdoms_from_religion = {}
    for line in config_file:
        split_line = [el.lower() for el in line.rstrip().split('#')[0].split('\t')]
        if len(split_line) == 2: #angle info
            empire, angle = split_line
            empires[empire] = Empire(name=empire, angle=angle)
        elif len(split_line) == 3: #terrain info
            empire, base_terrain, waste_terrain = split_line
            base_terrain_from_empire[empire] = Terrain[base_terrain]
            waste_terrain_from_empire[empire] = Terrain[waste_terrain]
            empires[empire].base_terrain = Terrain[base_terrain]
            empires[empire].waste_terrain = Terrain[waste_terrain]
        elif len(split_line) == 4: #kingdom info
            empire, kname, religion, geo_type = split_line
            assert geo_type == 'island' or geo_type == 'continent'
            kingdom = Kingdom(kname, geo_type == 'island', religion)
            empires[empire].add_kingdom(kingdom)
            if religion in kingdoms_from_religion:
                kingdoms_from_religion[religion].append(kingdom.name)
            else:
                kingdoms_from_religion[religion] = [kingdom.name]
    #TODO: Add some asserts that all the empires have the same keys?
    return (empires, kingdoms_from_religion, base_terrain_from_empire, waste_terrain_from_empire)


def find_ocean_and_wastelands(world):
    '''Given a tile, return a dictionary of distance from shore for the ocean and a list of list of hexes for each interior hole.'''
    bounding_hex = continent_gen.BoundingHex(world, extra = 1)
    non_land = [k for k, v in bounding_hex.dists.items() if v >= 1]
    non_land_chunks = continent_gen.get_chunks(non_land)
    max_index = -1
    max_len = 0
    for idx in range(len(non_land_chunks)):
        if len(non_land_chunks[idx]) > max_len:
            max_index = idx
            max_len = len(non_land_chunks[idx])
    ocean = non_land_chunks.pop(max_index)
    return {k: bounding_hex.dists[k] for k in ocean}, non_land_chunks


def calc_land_height(world, ocean):
    land_height = {}
    boundary = world.boundary()
    height = 0
    next_set = set()
    for el in boundary:
        if any([nel in ocean for nel in el.neighbors()]):
            land_height[el] = height
            next_set.update([nel for nel in el.neighbors() if nel not in ocean])
    current = set(next_set)
    while len(current) > 0:
        height += 1
        next_set = set()
        for el in current:
            if el not in land_height:
                land_height[el] = height
            next_set.update([nel for nel in el.neighbors() if nel not in ocean and nel not in land_height])
        current = set(next_set)
    return land_height


def check_contiguous(group):
    '''Check to make sure that every cube in a list is reachable by walking through that list.'''
    if len(group) == 0:
        return True
    to_search = {group[0]}
    found = set()
    nfound = 0
    while len(to_search) > 0:
        curr = to_search.pop()
        nfound += 1
        found.add(curr)
        try:
            to_search.union([el for el in curr.neighbors() if el in group and el not in to_search and el not in found])
        except:
            continue
    return len(group) == nfound


def split_group(divide):
    '''Divide a group into two groups, both about half the size.'''
    target_low = math.floor(len(divide) * 0.5)
    target_high = math.ceil(len(divide) * 0.5)
    maxnn = 0
    for el in divide:
        nn = 0
        for nel in el.neighbors():
            if nel not in divide:
                nn += 1
        if nn > maxnn:
            maxnn = nn
            loc = el
    valid = False
    locs_to_try = random.sample(divide, len(divide))
    try_index = 0
    while not valid and try_index < len(divide):
        dist_to_loc = {}
        to_fill_el = [loc]
        to_fill_d = [0]
        while len(to_fill_el)>0:
            curr = to_fill_el.pop(0)
            dist = to_fill_d.pop(0)
            valid_neighbors = [el for el in curr.neighbors() if el not in dist_to_loc and el not in to_fill_el and el in divide]
            to_fill_el.extend(valid_neighbors)
            to_fill_d.extend([dist+1]*len(valid_neighbors))
            dist_to_loc[curr] = dist
        sorted_dtl = sorted(dist_to_loc.items(), key=lambda x: x[1])
        if sorted_dtl[target_low-1][1] < sorted_dtl[target_low][1]:
            a = [el[0] for el in sorted_dtl[:target_high]]
            b = [el[0] for el in sorted_dtl[target_high:]]
        else:
            median = sorted_dtl[target_low][1]
            a = [el[0] for el in sorted_dtl if el[1] < median]
            b = [el[0] for el in sorted_dtl if el[1] > median]
            m = [el[0] for el in sorted_dtl if el[1] == median]
            while len(m) > 0:
                na = []
                nb = []
                nm = []
                for el in m:
                    n = el.neighbors()
                    if len([el for el in n if el in a]) > len([el for el in n if el in b]):
                        na.append(el)
                    elif len([el for el in n if el in a]) < len([el for el in n if el in b]):
                        nb.append(el)
                    else:
                        nm.append(el)
                if len(na) + len(nb) == 0:
                    random.shuffle(m)
                    if len(a) < len(b): #Favor a if it has less elements, otherwise b.
                        a.extend(m[:math.ceil(len(m)*0.5)])
                        b.extend(m[math.ceil(len(m)*0.5):])
                    else:
                        b.extend(m[:math.ceil(len(m)*0.5)])
                        a.extend(m[math.ceil(len(m)*0.5):])
                    m = []
                else:
                    a.extend(na)
                    b.extend(nb)
                    m = nm
        #Check that a and b are contiguous.
        valid = check_contiguous(a) and check_contiguous(b) and len(a) > 0 and len(b) > 0
        if not valid:
            loc = locs_to_try[try_index]
            try_index += 1
    if not valid: #We tried all of the possible starting locations and couldn't get an even contiguous split.
        maxnn = 0
        for el in divide:
            nn = 0
            for nel in el.neighbors():
                if nel in divide:
                    nn += 1
            if nn > maxnn:
                maxnn = nn
                loc = el
        #Now we have the element with the most neighbors in the set, which means it's most likely the important boundary.
        #We can build subsets in a way that guarantees contiguity.
        sources = [el for el in loc.neighbors() if el in divide]
        groups = []
        while len(sources) > 0:
            source = sources.pop(0)
            if not any([source in group for group in groups]):
                groups.append([source])
                m = [el for el in source.neighbors() if el in divide and el != loc]
                while len(m) > 0:
                    curr = m.pop(0)
                    if curr not in groups[-1]:
                        groups[-1].append(curr)
                    m.extend([el for el in curr.neighbors() if el in divide and el not in groups[-1] and el != loc])
        max_len = max([len(group) for group in groups])
        groups = [group for group in groups if len(group) == max_len]
        a = random.sample(groups,1)[0]
        b = [el for el in divide if el not in a]
    return (a,b)


def group_seas(ocean, min_sea_size=3, max_sea_size=12):
    '''Given a dictionary of distance from shore, create a list of list of hexes that are shallow water,
    and a list of list of hexes that are deeper (but still traversable) water.'''
    water_groups = []
    shore = [k for k, v in ocean.items() if v == 1]
    shore_n = [len([nel for nel in el.neighbors() if nel in shore]) for el in shore]
    w_n = [len([nel for nel in el.neighbors() if nel in ocean]) for el in shore]
    eshore = [el for index, el in enumerate(shore) if shore_n[index] == w_n[index]]
    for el in eshore:
        if not any([el in wg for wg in water_groups]):
            water_groups.append([el])
            to_search = [h for h in el.neighbors() if h in eshore]
            while len(to_search) > 0:
                curr = to_search.pop()
                if curr not in water_groups[-1]:
                    water_groups[-1].append(curr)
                to_search.extend([h for h in curr.neighbors() if h in eshore and h not in water_groups[-1]])
    to_split = [wg for wg in water_groups if len(wg) > max_sea_size]
    while len(to_split) > 0:
        divide = to_split[0]
        a, b = split_group(divide)
        water_groups.remove(divide)
        water_groups.append(a)
        water_groups.append(b)
        to_split = [wg for wg in water_groups if len(wg) > max_sea_size]
    water_groups = [wg for wg in water_groups if len(wg) > 1]
    for wg in water_groups:
        if len(wg) <= min_sea_size:
            neighbors = []
            for el in wg:
                neighbors.extend([nel for nel in el.neighbors() if nel in shore and not any([nel in wag for wag in water_groups])])
            neighbors = list(set(neighbors))
            if len(wg) + len(neighbors) <= max_sea_size:
                wg.extend(neighbors)
    ends = [el for el in shore if not any([el in wag for wag in water_groups]) and 
        sum([nel in shore and not any([nel in wag for wag in water_groups]) for nel in el.neighbors()]) == 1]
    while len(ends) > 0:
        for src in ends:
            if not any(src in wag for wag in water_groups):
                water_groups.append([src])
                valid = True
                while valid:
                    neighbors = []
                    for el in water_groups[-1]:
                        neighbors.extend([nel for nel in el.neighbors() if nel in shore and not any([nel in wag for wag in water_groups])])
                    if len(neighbors) == 0:
                        valid = False
                        if len(water_groups[-1]) == 1: #We should just glom on to a neighbor.
                            bn_size = 0
                            bn_ind = 0
                            for index, wg in enumerate(water_groups):
                                n_size = sum([nel in wg for nel in src.neighbors()])
                                if n_size > bn_size:
                                    bn_ind = index
                            water_groups[bn_ind].append(src)
                            water_groups.pop()
                    elif len(water_groups[-1]) + len(neighbors) < max_sea_size:
                        water_groups[-1].extend(neighbors)
                    else:
                        valid = False
        ends = [el for el in shore if not any([el in wag for wag in water_groups]) and 
            sum([nel in shore and not any([nel in wag for wag in water_groups]) for nel in el.neighbors()]) == 1]
        lonely_shore = [el for el in shore if not any([el in wag for wag in water_groups])]
        if len(ends) == 0 and len(lonely_shore) > 0:
            ends = random.sample(lonely_shore,1)
    depths = [el for el, h in ocean.items() if h == 1]
    seas = []
    for el in depths:
        nels = el.neighbors()
        if all([nel in ocean and ocean[nel] == 0 for nel in nels]):
            most_adj = 0
            for wg in water_groups:
                adj = sum([nel in wg for nel in nels])
                if adj > 0 and len(wg) == 2:
                    best = wg
                    most_adj = 7
                elif adj > most_adj:
                    best = wg
                    most_adj = adj
            best.append(el)
        elif all([nel in ocean and ocean[nel] < 2 for nel in nels]):
            seas.append(el)
    sea_groups = []
    for el in seas:
        if not any(el in sg for sg in sea_groups):
            sea_groups.append([el])
            to_search = [h for h in el.neighbors() if h in seas]
            while len(to_search) > 0:
                curr = to_search.pop()
                if curr not in sea_groups[-1]:
                    sea_groups[-1].append(curr)
                to_search.extend([h for h in curr.neighbors() if h in seas and h not in sea_groups[-1]])
    for sg in sea_groups:
        if len(sg) <= 2:
            for el in sg:
                most_adj = 0
                nels = el.neighbors()
                for wg in water_groups:
                    adj = sum([nel in wg for nel in nels])
                    if adj > 0 and len(wg) == 2:
                        best = wg
                        most_adj = 7
                    elif adj > most_adj:
                        best = wg
                        most_adj = adj
                best.append(el)
    sea_groups = [sg for sg in sea_groups if len(sg) > 2]
    if len(sea_groups) > 0:
        while max([len(el) for el in sea_groups]) > 5:
            to_split = [sg for sg in sea_groups if len(sg) > 5]
            divide = to_split[0]
            a, b = split_group(divide)
            sea_groups.remove(divide)
            sea_groups.append(a)
            sea_groups.append(b)
    return water_groups, sea_groups


def new_rgb(dictionary):
    '''Returns a randomly chosen RGB value that isn't already in the values of dictionary.'''
    guess = tuple(np.random.randint(0,256,3))
    if guess in dictionary.values() or guess == (255,255,255) or guess == (0,0,0):
        return new_rgb(dictionary)
    else:
        return guess

def allocate_pids(world, wastelands, shore_groups, sea_groups):
    '''Assigns a pid and RGB to each hex.'''
    pid_from_hex = {}
    rgb_from_hex = {}
    rgb_from_pid = {}
    last_pid = 1
    for el in world.real_hex_list():
        rgb = new_rgb(rgb_from_pid)
        rgb_from_hex[el] = rgb
        pid_from_hex[el] = last_pid
        rgb_from_pid[last_pid] = rgb
        last_pid += 1
    for group in wastelands:
        rgb = new_rgb(rgb_from_pid)
        rgb_from_pid[last_pid] = rgb
        for el in group:
            rgb_from_hex[el] = rgb
            pid_from_hex[el] = last_pid
        last_pid += 1
    for group in shore_groups + sea_groups:
        rgb = new_rgb(rgb_from_pid)
        rgb_from_pid[last_pid] = rgb
        for el in group:
            rgb_from_hex[el] = rgb
            pid_from_hex[el] = last_pid
        last_pid += 1
    return pid_from_hex, rgb_from_hex, rgb_from_pid


def make_terrain(world, wastelands, ocean, empires, base_terrain_from_empire, waste_terrain_from_empire):
    '''Create the dictionary that maps cubes to terrain.
    The main terrain used is:
        plains: the default, present everywhere
        farmland: rare, present everywhere, biased towards capitals.
        mountains: rare, present everywhere
        hills: uncommon, present everywhere

    There are four regional variants for wasteland terrain:
        forest (western Europe)
        steppe (eastern Europe)
        jungle (india)
        desert (islam)
    '''
    terrain_from_hex = {}
    flattened_kingdom_hexes = [item for sublist in continent_gen.KINGDOM_SIZE_LIST for item in sublist]
    for emp_idx, empire in enumerate(empires.keys()):
        base_terrain = base_terrain_from_empire[empire]
        waste_terrain = waste_terrain_from_empire[empire]
        for tile in world.tile_list[emp_idx].tile_list:
            tile_hexes = tile.real_hex_list()
            if len(tile_hexes) == NUM_KINGDOM_HEXES:
                # There are 16 counties: 1 capital farmland, 8 plains, 4 hills, and 3 mountains.
                terrain_list = [Terrain.plains] * 6 + [Terrain.hills] * 4 + [Terrain.mountains] * 3 + [base_terrain] * 2
                random.shuffle(terrain_list)
                terrain_list.insert(0, Terrain.farmlands)
                terrain_list = [[terrain_type] * num_hexes for terrain_type, num_hexes in zip(terrain_list, flattened_kingdom_hexes)]
            elif len(tile_hexes) == NUM_CENTER_HEXES:
                # There are 4 counties: farmland capital, 2 plains, and 1 hills.
                terrain_list = [Terrain.plains] * 2 + [Terrain.hills] * 1
                random.shuffle(terrain_list)
                terrain_list.insert(0, Terrain.farmlands)
                terrain_list = [[terrain_type] * num_hexes for terrain_type, num_hexes in zip(terrain_list, continent_gen.CENTER_SIZE_LIST)]
            elif len(tile_hexes) == NUM_BORDER_HEXES:
                # There are 3 counties: 1 plains, 1 hills, and 1 mountains.
                terrain_list = [Terrain.plains] * 1 + [Terrain.hills] * 1 + [Terrain.mountains] * 1
                random.shuffle(terrain_list)
                terrain_list = [[terrain_type] * num_hexes for terrain_type, num_hexes in zip(terrain_list, continent_gen.BORDER_SIZE_LIST)]
            else:
                raise ValueError('Don\'t know how to handle terrain for this tile: {}'.format(tile))
            # This was a list of lists, where each barony in a county just shares the same terrain, and we want it to be a flattened list to match the real_hex_list.
            terrain_list = [item for sublist in terrain_list for item in sublist]
            for el in tile_hexes:
                terrain_from_hex[el] = terrain_list.pop(0)
            bounding_hex = continent_gen.BoundingHex(world.tile_list[emp_idx])
            for wasteland in wastelands:
                if wasteland[0] in bounding_hex:
                    for el in wasteland:
                        terrain_from_hex[el] = waste_terrain
    return terrain_from_hex


def make_province_terrain_txt(terrain_from_hex, pid_from_hex, file_dir):
    '''Create the common/00_province_terrain.txt file.'''
    os.makedirs(os.path.join(file_dir,"common", "province_terrain"), exist_ok=True)
    with open(os.path.join(file_dir,"common", "province_terrain", "00_province_terrain.txt"),'w') as f:
        f.write('default=plains\n')
        for cube_loc, terrain_type in terrain_from_hex.items():
            f.write(f'{pid_from_hex[cube_loc]}={terrain_type.name}')


def small_from_empire(src_dir, ename, num, centers=False, used=[]):
    '''Reads in the list of possible duchies to use as centers and borders for an empire,
    and randomly selects the appropriate number.
    Won't select anything which has a substring in used.'''
    dtype = 'centers' if centers else 'borders'
    with open(os.path.join(src_dir, "common", "landed_titles",f"{ename}-{dtype}.txt")) as inf:
        poss_names = [line.strip() for line in inf.readlines()]
    for name in poss_names:
        if any(u in name for u in used):
            poss_names.remove(name)
    return poss_names


def title_order(empire, src_dir):
    '''Returns the names of files to write in the landed_titles, in order, from a list.'''
    num_kingdoms = empire.continentals
    num_centers = num_kingdoms - 2
    num_borders = 4 * num_kingdoms - 6
    centers = small_from_empire(src_dir, empire.name, num_centers, centers=True, used=empire.continental_names)
    borders = small_from_empire(src_dir, empire.name, num_borders, centers=False, used=empire.continental_names)
    kingdoms = [k for k in empire.continental_names]
    random.shuffle(centers)
    random.shuffle(borders)
    random.shuffle(kingdoms)
    titles = []
    titles.append(centers[0])
    titles.extend(kingdoms[:3])
    titles.extend(borders[:6])
    if num_kingdoms == 3:
        return titles
    titles.append(centers[1])
    titles.append(kingdoms[3])
    titles.extend(borders[6:10])
    if num_kingdoms == 4:
        return titles
    titles.append(centers[2])
    titles.append(kingdoms[4])
    titles.extend(borders[10:])
    return titles



def make_landed_titles(empires, src_dir, file_dir):
    '''Create the common/landed_titles/00_landed_titles.txt. Also generates the bname_from_pid dict.'''
    os.makedirs(os.path.join(file_dir,"common", "province_terrain"), exist_ok=True)
    bname_from_pid = {}
    pid=1
    with open(os.path.join(file_dir,"common", "province_terrain", "00_province_terrain.txt"),'w') as outf:
        # Initial constants.
        with open(os.path.join(src_dir, "common", "landed_titles","default.txt")) as inf:
            outf.write(inf.readlines())
        outf.write('\n')
        # Go through each empire, and then each kingdom in each empire.
        for ename, empire in empires.items():
            with open(os.path.join(src_dir, "common", "landed_titles", f"{ename}.txt")) as inf:
                outf.write(inf.readlines())
            titles = title_order(empire, src_dir)
            for title in titles:
                with open(os.path.join(src_dir, "common", "landed_titles", f"{title}.txt")) as inf:
                    for line in inf.readlines():
                        if write_pid:
                            line += str(pid)
                            pid += 1
                        if '\t\t\tb_' in line[:6]:
                            bname = line.split('b_')[1].split('=')[0].strip()
                            assert bname not in bname_from_pid
                            bname_from_pid[bname] = pid
                            write_pid = True
                        outf.write(line)
                outf.write('\t}\n')
            outf.write('}\n')

def make(file_dir = 'C:\\ck3_procedural\\', mod_name='testing_modgen', mod_disp_name='SinglePlayerTest',
         config_filepath = 'C:\\ck3_procedural\\config.txt',
         max_x=1280, max_y=1280, num_rivers = 25, crisp = True, seed = None):
    '''Attempts to make the folder with mod.'''
    #Basic Setup.
    mod_dir = os.path.join(file_dir,mod_name)
    if os.path.exists(mod_dir):
        shutil.rmtree(mod_dir)
    os.makedirs(mod_dir, exist_ok=True)
    os.makedirs(os.path.join(mod_dir,'map'), exist_ok=True)
    os.makedirs(os.path.join(mod_dir,'common'), exist_ok=True)
    make_dot_mod(file_dir, mod_name, mod_disp_name)
    if seed:
        random.seed(seed)
    #Read in the configuration files.
    empires, kingdoms_from_religion, base_terrain_from_empire, waste_terrain_from_empire = read_config_file(config_filepath)
    assert len(empires) == 3
    cont_size_list = [v.continentals for v in empires.values()]
    island_size_list = [v.islands for v in empires.values()]
    angles = [v.angle for v in empires.values()]
    #Generate the tile hierarchy.
    world = continent_gen.make_world(cont_size_list=cont_size_list, island_size_list=island_size_list, angles=angles)
    world.doodle_by_tile([(255,0,0),(0,255,0),(0,0,255)],size=(1500,1500))
    world.rectify()
    world.doodle_by_tile([(255,0,0),(0,255,0),(0,0,255)],size=(1500,1500))
    max_mag = max([el.mag() for el in world.real_hex_list()])
    #Fill in wastelands and group together the seas.
    ocean, wastelands = find_ocean_and_wastelands(world)
    shore_groups, sea_groups = group_seas(ocean)
    #Create hex to province id mapping and hex to RGB mapping.
    pid_from_hex, rgb_from_hex, rgb_from_pid = allocate_pids(world, wastelands, shore_groups, sea_groups)
    cmap = CK2Map(max_x, max_y, hex_size = 12, map_size = max_mag + 2, crisp=crisp, default = (255,255,255))
    cmap.d_cube2rgb = rgb_from_hex
    cmap.provinces(filedir=os.path.join(mod_dir,'map'))
    land_height = calc_land_height(world, ocean)
    cmap.d_cube2terr = make_terrain(world, wastelands, ocean, empires, base_terrain_from_empire, waste_terrain_from_empire)
    make_province_terrain_txt(cmap.d_cube2terr, pid_from_hex, mod_dir)
    waste_list = [*wastelands]
    name_from_pid = {k: str(k) for k in rgb_from_pid.keys()}
    make_definition(mod_dir, rgb_from_pid, name_from_pid)
    # cmap.topology(land_height, ocean, waste_list, filedir=os.path.join(mod_dir,"map"))
    # cmap.terrain(filedir=os.path.join(mod_dir,'map'))
    # rivers = make_rivers()
    # cmap.trees(self.land_height, self.water_height, waste_list, filedir=os.path.join(self.filedir,"map"))
    # cmap.rivers(self.land_height, filedir=os.path.join(self.filedir,"map"))
    return world, cmap, ocean, land_height, waste_list


def make_definition(file_dir, rgb_from_pid, name_from_pid):
    with open(os.path.join(file_dir,"map", "definition.csv"),'w') as f:
        f.write('0;0;0;0;x;x;\n')
        for pid, rgb in rgb_from_pid.items():
            f.write(';'.join([str(pid)] + [str(c) for c in rgb] + [name_from_pid[pid], 'x', '\n']))


if __name__ == '__main__':
    #Eventually we're going to use a config file to run most of this.
    make()
