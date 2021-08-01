from dataclasses import dataclass, field
from functools import lru_cache
from itertools import combinations
import math
import os
import random
import shutil
from typing import List

import numpy as np

from cube import Cube
from tile import Tile
from ck3map import CK3Map, IMPASSABLE, new_rgb, Terrain
import continent_gen

NUM_KINGDOM_HEXES = sum([sum(x) for x in continent_gen.KINGDOM_SIZE_LIST])
NUM_CENTER_HEXES = sum(continent_gen.CENTER_SIZE_LIST)
NUM_BORDER_HEXES = sum(continent_gen.BORDER_SIZE_LIST)

RELIGION_DICT = {'catholic': 'christianity', 'orthodox': 'christianity', 'islam': 'islam'}

# Maybe should parse this out of the base game file instead? That'd make it update more easily.
HOLY_SITE_LIST = [
    "jerusalem", "rome", "cologne", "santiago", "kent", "segrada_familia", "constantinople", "alexandria", "antioch", "aksum", "napata",
    "suenik", "colonea", "beirut", "baghdad", "farz", "kerala", "armagh", "iona", "visoki", "ragusa", "esztergom", "mecca", "medina",
    "cordoba", "sinai", "tinmallal", "fes", "nadjaf", "damascus", "siffa", "kufa", "basra", "nizwa", "sijilmasa", "bahrein", "yamama",
    "sinjar", "baalbek", "lalish", "nishapur", "dashtestan", "zozan", "semien", "sufed", "gerizim", "ahvaz", "samarkand", "udabhanda",
    "toledo", "memphis", "bodh_gaya", "kusinagara", "sarnath", "sanchi", "ajanta", "sagya", "dagon", "pagan", "atamasthana",
    "amaravati", "varanasi", "ayodhya", "mathura", "haridwar", "kanchipuram", "ujjayini", "dwarka", "palitana", "shikharj",
    "ranakpur", "ellora", "sittannavasal", "itanagar", "pemako", "rima", "tezu", "kathmandu", "ilam", "garwhal", "jumla", "maowun",
    "ngawa", "chakla", "gyaitang", "rebgong", "xingqing", "yijinai", "alxa", "yazd", "nok_kundi", "takht-i-sangin",
    "takht-e_soleyman", "ushi-darena", "uppsala", "lejre", "paderborn", "zeeland", "ranaheim", "jorvik", "konugardr", "raivola",
    "hiiumaa", "akkel", "perm", "kiev", "novgorod", "barlad", "plock", "pokaini", "torun", "braslau", "rugen", "pest", "kerch",
    "olvia", "poszony", "sarysyn", "kara_khorum", "qayaliq", "tavan_bogd", "preslav", "athens", "carthage", "olympus", "sparta",
    "kabul", "multan", "bost", "khotan", "balkh", "lhasa", "purang", "tyumen", "surgut", "ob", "olkhon", "awkar", "jenne",
    "niani", "kukiya", "wadan", "daura", "garumele", "igbo", "el_fasher", "wandala", "kisi", "sherbro", "kayor", "kasa",
    "bono", "kumasi", "ife", "nikki", "aswan", "wadi_el_milk", "naqis", "sennar", "danakil", "kaffa", "harar", "makhir",
    "mogadishu", "gilgit", "suzhou"
]

def make_dot_mod(file_dir, mod_name, mod_disp_name):
    '''Build the basic mod details file.'''
    shared = "version = \"0.0.1\"\n"
    shared += "tags = {\n\t\"Total Conversion\"\n}\n"
    shared += "name = \"{}\"\n".format(mod_disp_name)
    shared += "supported_version = \"1.4.4\"\n"
    outer = "path = \"mod/{}\"\n".format(mod_name)
    
    replace_paths = [
        "common/bookmark_portraits", "common/culture/innovations", "common/dynasties",
        "history/characters", "history/cultures", "history/province_mappings", "history/provinces", "history/titles", "history/wars"
        ]
    shared += "replace_path = \"" + "\"\nreplace_path = \"".join(replace_paths)+"\""
    os.makedirs(os.path.join(file_dir, mod_name), exist_ok=True)
    with open(os.path.join(file_dir,"{}.mod".format(mod_name)),'w') as f:
        f.write(shared + outer)
    with open(os.path.join(file_dir, mod_name, "descriptor.mod".format(mod_name)),'w') as f:
        f.write(shared)


@dataclass
class MinorIsland:
    name: str = ""
    religion: str = ""

@dataclass
class Center:
    name: str = ""
    religion: str = ""

@dataclass
class Border:
    name: str = ""
    religion: str = ""

@dataclass
class Kingdom:
    name: str = ""
    island: bool = False
    religion: str = ""


@dataclass
class Empire:
    name: str = ""
    angle: int = 0
    kingdoms: List["Kingdom"] = field(default_factory=list)
    centers: List["Center"] = field(default_factory=list)
    borders: List["Border"] = field(default_factory=list)
    minor_islands: List["MinorIsland"] = field(default_factory=list)
    base_terrain: Terrain = Terrain.plains
    waste_terrain: Terrain = Terrain.plains
    continentals: int = 0
    islands: int = 0
    titles: List = field(default_factory=list)
    
    def title_order(self):
        '''Returns the names of files to write in the landed_titles, in order, from a list.'''
        if len(self.titles) > 0:
            return self.titles
        num_kingdoms = self.continentals
        num_centers = num_kingdoms - 2
        num_borders = 2 * num_kingdoms - 3
        assert len(self.centers) >= num_centers, 'Not enough centers for {}'.format(self.name)
        assert len(self.borders) >= num_borders, 'Not enough borders for {}'.format(self.name)
        centers = [c.name for c in self.centers]
        random.shuffle(centers)
        centers = centers[:num_centers]
        borders = [c.name for c in self.borders]
        random.shuffle(borders)
        borders = borders[:num_borders]
        kingdoms = [k for k in self.continental_names()]
        island_kingdoms = [k for k in self.island_names()]
        random.shuffle(kingdoms)
        random.shuffle(island_kingdoms)
        titles = []
        titles.append(centers[0])
        titles.extend(kingdoms[:3])
        titles.extend(borders[:3])
        if num_kingdoms == 3:
            titles.extend(island_kingdoms)
            return titles
        titles.append(centers[1])
        titles.append(kingdoms[3])
        titles.extend(borders[3:5])
        if num_kingdoms == 4:
            titles.extend(island_kingdoms)
            return titles
        titles.append(centers[2])
        titles.append(kingdoms[4])
        titles.extend(borders[5:])
        titles.extend(island_kingdoms)
        return titles

    def add_kingdom(self, kingdom):
        self.kingdoms.append(kingdom)
        if kingdom.island:
            self.islands += 1
        else:
            self.continentals += 1

    def continental_names(self):
        return [k.name for k in self.kingdoms if not k.island]

    def island_names(self):
        return [k.name for k in self.kingdoms if k.island]

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
    others_from_religion = {}
    for line in config_file:
        split_line = [el.lower() for el in line.rstrip().split('#')[0].split('\t')]
        if len(split_line) == 2: #angle info
            empire, angle = split_line
            empires[empire] = Empire(name=empire, angle=int(angle))
        elif len(split_line) == 3: #terrain info
            empire, base_terrain, waste_terrain = split_line
            base_terrain_from_empire[empire] = Terrain[base_terrain]
            waste_terrain_from_empire[empire] = Terrain[waste_terrain]
            empires[empire].base_terrain = Terrain[base_terrain]
            empires[empire].waste_terrain = Terrain[waste_terrain]
        elif len(split_line) == 4: #kingdom info
            empire, kname, religion, geo_type = split_line
            assert geo_type == 'island' or geo_type == 'continent'
            if kname[0] == 'k':
                kingdom = Kingdom(kname, geo_type == 'island', religion)
                empires[empire].add_kingdom(kingdom)
                if religion in kingdoms_from_religion:
                    kingdoms_from_religion[religion].append(kingdom.name)
                else:
                    kingdoms_from_religion[religion] = [kingdom.name]
            elif kname[0] == 'c':
                center = Center(kname, religion)
                empires[empire].centers.append(center)
                if religion in others_from_religion:
                    others_from_religion[religion].append(center.name)
                else:
                    others_from_religion[religion] = [center.name]
            elif kname[0] == 'b':
                border = Border(kname, religion)
                empires[empire].borders.append(border)
                if religion in others_from_religion:
                    others_from_religion[religion].append(border.name)
                else:
                    others_from_religion[religion] = [border.name]
            elif kname[0] == 'm':
                minor_island = MinorIsland(kname, religion)
                empires[empire].minor_islands.append(minor_island)
                if religion in others_from_religion:
                    others_from_religion[religion].append(minor_island.name)
                else:
                    others_from_religion[religion] = [minor_island.name]
    return (empires, kingdoms_from_religion, others_from_religion, base_terrain_from_empire, waste_terrain_from_empire)


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
    return pid_from_hex, rgb_from_hex, rgb_from_pid, last_pid


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
    with open(os.path.join(file_dir,"common", "province_terrain", "00_province_terrain.txt"),'w', encoding='utf8') as f:
        f.write('default=plains\n')
        buffer = []
        for cube_loc, terrain_type in terrain_from_hex.items():
            buffer.append((pid_from_hex[cube_loc], terrain_type.name))
        buffer = sorted(buffer)
        f.write("\n".join([str(a) + "=" + b for a, b in buffer]) + "\n")
    with open(os.path.join(file_dir,"common", "province_terrain", "01_province_properties.txt"),'w', encoding='utf8') as f:
        # This is where winter severity would go.
        f.write("\n")


def make_landed_titles(empires, religions, src_dir, out_dir):
    '''Create the common/landed_titles/00_landed_titles.txt (and map_data/geographical_region). Also generates the bname_from_pid dict.'''
    os.makedirs(os.path.join(out_dir,"common", "landed_titles"), exist_ok=True)
    os.makedirs(os.path.join(out_dir,"history", "provinces"), exist_ok=True)
    bname_from_pid = {}
    bname = "ERROR"
    pid=1
    with open(os.path.join(out_dir, "map_data", "island_region.txt"), "w", encoding='utf8') as isl_region_file:
        # TODO: Make this real.
        isl_region_file.write("#No islands generated.\n")
    with open(os.path.join(out_dir, "common", "landed_titles", "00_landed_titles.txt"),'w', encoding='utf8') as outf:
        with open(os.path.join(out_dir, "map_data", "geographical_region.txt"), "w", encoding='utf8') as geo_region_file:
            # Initial constants.
            with open(os.path.join(src_dir, "start", "common", "landed_titles", "landed_titles.txt"), encoding='utf8') as inf:
                outf.write(inf.read())
            outf.write('\n')
            # Go through each religion to write out the special titles.
            for religion in religions:
                with open(os.path.join(src_dir, "r_" + religion, "common", "landed_titles", "landed_titles.txt"), encoding='utf8') as inf:
                    outf.write(inf.read())
                outf.write('\n')
            outf.write('\n')
            # Go through each empire, and then each kingdom in each empire.
            for ename, titles in empires.items():
                with open(os.path.join(src_dir, "e_" + ename, "common", "landed_titles", "landed_titles.txt"), encoding='utf8') as inf:
                    # If you want to edit the empire capital, this is where to do it.
                    outf.write(inf.read())
                outf.write('\n')
                geo_region_duchy_list = []
                geo_region_kingdom_list = []
                kingdom_duchy_list = []
                # Sadly, the title_order is the order in which the hexes are placed (and thus the pids are calculated),
                # but it's not the order in which the titles are written, because we need to lump all of the duchies
                # together under a dummy kingdom. So the duchies go first, and then the kingdoms.
                kingdom_buffer = ""
                for title in titles:
                    barony_mapping = {}
                    if title[0] == 'k':
                        # For kingdoms, we want to construct a geographical region in the georegion file.
                        # TODO: This is where we should check for island kingdoms and add them to island_region instead.
                        region_name = f"world_{ename}_{title}"
                        kingdom_duchy_list = []
                        geo_region_kingdom_list.append(region_name)
                    with open(os.path.join(src_dir, title, "common", "landed_titles", "landed_titles.txt"), encoding='utf8') as inf:
                        for line in inf.readlines():
                            if line.startswith("\t\td_"):
                                if title[0] == "k":
                                    kingdom_duchy_list.append(line.split("=")[0].strip())
                                else:
                                    geo_region_duchy_list.append(line.split("=")[0].strip())
                            if "\t\tb_" in line:
                                bname = line.split('\t\tb_')[1].split('=')[0].strip()
                            if "province = P" in line:
                                front, back = line.split("= P")
                                barony_mapping["P" + back.split()[0]] = str(pid)
                                line = front + "= " + str(pid) + "\n"
                                bname_from_pid[pid] = bname
                                pid += 1
                            if title[0] == 'k':
                                kingdom_buffer += line
                            else:
                                outf.write(line)
                    if title[0] == 'k':
                        # Write the kingdom's geographical region.
                        duchies = " ".join(kingdom_duchy_list)
                        geo_region_file.write(f"{region_name} = {{\n\tduchies = {{\n\t\t{duchies}\t}}\n}}\n")
                        kingdom_buffer += "\n"
                    else:
                        outf.write("\n")
                    # Write out history/provinces file.
                    with open(os.path.join(src_dir, title, "history", "provinces", title + ".txt"), encoding='utf8') as inf:
                        with open(os.path.join(out_dir, "history", "provinces", title + ".txt"), 'w', encoding='utf8') as hpoutf:
                            for line in inf.readlines():
                                if line[0] == 'P':
                                    sloc = line.index(" ")
                                    try:
                                        line = barony_mapping[line[:sloc]] + line[sloc:]
                                    except:
                                        raise(Exception(f"{line} {sloc} not found in barony_mapping (len({len(barony_mapping)})), {title}."))
                                hpoutf.write(line)

                outf.write("\t}\n") # End the dummy kingdom for all of the duchies.
                outf.write(kingdom_buffer)
                outf.write('\n}\n')
                # Write the middle geographical region.
                middle_region_name = f"world_{ename}_middle"
                duchies = " ".join(geo_region_duchy_list)
                geo_region_file.write(f"{middle_region_name} = {{\n\tduchies = {{\n\t\t{duchies}\t}}\n}}\n")
                # Write the empire's geographical region.
                regions = " ".join(geo_region_kingdom_list + [middle_region_name])
                geo_region_file.write(f"world_{ename} = {{\n\tregions = {{\n\t\t{regions}\t}}\n}}\n")
    return bname_from_pid

def make(file_dir = 'C:\\ck3_procedural\\', mod_name='testing_modgen', mod_disp_name='SinglePlayerTest',
         config_filepath = 'C:\\ck3_procedural\\config.txt',
         max_x=1280, max_y=1280, num_rivers = 25, crisp = True, seed = None):
    '''Attempts to make the folder with mod.'''
    #Basic Setup.
    mod_dir = os.path.join(file_dir, mod_name)
    data_dir = os.path.join(file_dir, "data")
    if os.path.exists(mod_dir):
        shutil.rmtree(mod_dir)
    os.makedirs(mod_dir, exist_ok=True)
    os.makedirs(os.path.join(mod_dir,"map_data"), exist_ok=True)
    os.makedirs(os.path.join(mod_dir,'common'), exist_ok=True)
    make_dot_mod(file_dir, mod_name, mod_disp_name)
    if seed:
        random.seed(seed)
    #Read in the configuration files.
    empires, kingdoms_from_religion, others_from_religion, base_terrain_from_empire, waste_terrain_from_empire = read_config_file(config_filepath)
    e_titles = {k: v.title_order() for k, v in empires.items()}
    religions = make_religions(e_titles, "c_roma", data_dir, mod_dir)  # TODO: update the default hs_county to be read from the list of real counties.
    assert len(empires) == 3
    cont_size_list = [v.continentals for v in empires.values()]
    island_size_list = [v.islands for v in empires.values()]
    assert(sum(island_size_list)) == 0, "Haven't implemented all the island stuff yet."
    angles = [v.angle for v in empires.values()]
    #Generate the tile hierarchy.
    world = continent_gen.make_world(cont_size_list=cont_size_list, island_size_list=island_size_list, angles=angles)
    world.rectify()
    max_mag = max([el.mag() for el in world.real_hex_list()])
    #Fill in wastelands and group together the seas.
    ocean, wastelands = find_ocean_and_wastelands(world)
    shore_groups, sea_groups = group_seas(ocean)
    #Create hex to province id mapping and hex to RGB mapping.
    pid_from_hex, rgb_from_hex, rgb_from_pid, last_pid = allocate_pids(world, wastelands, shore_groups, sea_groups)
    cmap = CK3Map(max_x, max_y, hex_size = 12, map_size = max_mag + 2, crisp=crisp, default = IMPASSABLE)
    cmap.d_cube2rgb = rgb_from_hex
    cmap.d_cube2pid = pid_from_hex
    cmap.provinces(filedir=mod_dir)
    land_height = calc_land_height(world, ocean)
    cmap.d_cube2terr = make_terrain(world, wastelands, ocean, e_titles, base_terrain_from_empire, waste_terrain_from_empire)
    make_province_terrain_txt(cmap.d_cube2terr, pid_from_hex, mod_dir)
    waste_list = [*wastelands]
    bname_from_pid = make_landed_titles(e_titles, religions, data_dir, mod_dir)
    # Make a bunch of map_data files. Maybe this should be split out to a separate function?
    make_adjacencies(mod_dir, world, pid_from_hex, bname_from_pid, cmap)
    make_climate(mod_dir)
    make_definition(mod_dir, rgb_from_pid, bname_from_pid)
    cmap.heightmap(land_height, ocean, waste_list, filedir=mod_dir)
    major_rivers, last_pid = cmap.rivers(land_height, bname_from_pid, num_rivers, last_pid, mod_dir)
    cmap.positions(bname_from_pid, filedir=mod_dir)
    make_dynasties(e_titles, religions, data_dir, mod_dir)
    make_default(mod_dir, waste_list, pid_from_hex, major_rivers, last_pid)
    make_bookmark(data_dir, mod_dir)
    make_innovations(data_dir, mod_dir)
    for category in [["events"]] + [["common", x] for x in ["casus_belli_types", "decisions", "flavorization", "on_action", "story_cycles"]]:
        copy_over(os.path.join(data_dir, "start"), mod_dir, so_far=category)
    return world, cmap, ocean, land_height, waste_list, bname_from_pid, last_pid


def make_adjacencies(file_dir, world, pid_from_hex, bname_from_pid, cmap):
    with open(os.path.join(file_dir,"map_data", "adjacencies.csv"),'w', encoding='utf8') as f:
        f.write('From;To;Type;Through;start_x;start_y;stop_x;stop_y;Comment\n')
        # Check for Gibraltar-like adjacencies.
        for (tile_a, tile_b) in combinations(world.tile_list, 2):
            strait_pairs, corner_pairs = tile_a.two_step_pairs(tile_b)
            # Exclude any corner pairs that are next to a strait pair, which should be used instead.
            corner_pairs = [c for c in corner_pairs if not(any([c[0] in s[0].neighbors() or c[1] in s[1].neighbors() for s in strait_pairs]))]
            for (hex_a, hex_b) in strait_pairs:
                # This is a connection between the two vertices of a strait, calculated from two trios of hexes.
                # TODO: Probably this logic should live in the ck3map class, and just return the x,y pairs?
                pid_a = pid_from_hex[hex_a]
                pid_b = pid_from_hex[hex_b]
                trios = hex_a.foursome(hex_b)
                x_a, y_a = cmap.find_vertex(trios[0])
                x_b, y_b = cmap.find_vertex(trios[1])
                comment = f"{bname_from_pid[pid_a]}-{bname_from_pid[pid_b]}"
                # From;To;Type;Through;start_x;start_y;stop_x;stop_y;Comment
                f.write(";".join([str(s) for s in [pid_a, pid_b, "sea", x_a, y_a, x_b, y_b, comment]]) + "\n")
            for (hex_a, hex_b) in corner_pairs:
                # This is a connection between the two edges separated by one water hex.
                pid_a = pid_from_hex[hex_a]
                pid_b = pid_from_hex[hex_b]
                middle = hex_a.avg(hex_b)
                x_a, y_a = cmap.edge_middle(hex_a, middle)
                x_b, y_b = cmap.edge_middle(hex_b, middle)
                comment = f"{bname_from_pid[pid_a]}-{bname_from_pid[pid_b]}"
                # From;To;Type;Through;start_x;start_y;stop_x;stop_y;Comment
                f.write(";".join([str(s) for s in [pid_a, pid_b, "sea", x_a, y_a, x_b, y_b, comment]]) + "\n")
        # TODO: Add adjacency for island kingdoms.
        
        # This line is now written later, because maybe you added major river adjacencies.
        # f.write('-1;-1;;-1;-1;-1;-1;-1;')


def make_bookmark(src_dir, out_dir):
    '''Makes the bookmarks.'''
    # TODO: Make this actually dynamic.
    os.makedirs(os.path.join(out_dir,"common", "bookmarks"), exist_ok=True)
    with open(os.path.join(out_dir,"common", "bookmarks", "00_bookmarks.txt"),'w', encoding='utf8') as outf:
        outf.write("mp_game = {\n")
        outf.write("\tstart_date = 1000.1.1\n")
        outf.write("\tis_playable = yes\n\n")
        outf.write("\tcharacter = {\n")
        outf.write("\t\tname = \"Hugues\"\n")
        outf.write("\t\tdynasty = 440\n")
        outf.write("\t\tdynasty_splendor_level = 1\n")
        outf.write("\t\ttype = male\n")
        outf.write("\t\tbirth = 980.11.26\n")
        outf.write("\t\ttitle = k_france\n")
        outf.write("\t\tgovernment = feudal_government\n")
        outf.write("\t\tculture = french\n")
        outf.write("\t\treligion = catholic\n")
        outf.write("\t\tdifficulty = \"BOOKMARK_CHARACTER_DIFFICULTY_EASY\"\n")
        outf.write("\t\thistory_id = 2202\n")
        outf.write("\t\tposition = { 765 590 }\n\n")
        outf.write("\t\tanimation = disapproval\n")
        outf.write("\t}\n}\n")
    os.makedirs(os.path.join(out_dir, "common", "bookmark_portraits"), exist_ok=True)
    with open(os.path.join(out_dir, "common", "bookmark_portraits", "bookmark_test_character.txt"),'w', encoding='utf8') as outf:
        with open(os.path.join(src_dir, "start", "common", "bookmark_portraits", "bookmark_test_character.txt"), encoding='utf8') as inf:
            outf.write(inf.read())       


def make_climate(file_dir):
    with open(os.path.join(file_dir,"map_data", "climate.txt"),'w', encoding='utf8') as f:
        # TODO: This is currently just a placeholder climate file (borrowed from https://forum.paradoxplaza.com/forum/threads/resource-nearly-blank-map-a-starting-point-for-tcs-v1-2.1414703/page-5#post-27133344 ). 
        f.write('mild_winter = {\n\t1 2 3 4 5 6 7 8\n}\n')


def make_default(file_dir, waste_list, pid_from_hex, major_rivers, last_pid):
    with open(os.path.join(file_dir,"map_data", "default.map"),'w', encoding='utf8') as f:
        f.write("definitions = \"definition.csv\"\n")
        f.write("provinces = \"provinces.png\"\n")
        f.write("rivers = \"rivers.png\"\n")
        f.write("topology = \"heightmap.heightmap\"\n")
        f.write("continent = \"continent.txt\"\n")
        f.write("adjacencies = \"adjacencies.csv\"\n")
        f.write("island_region = \"island_region.txt\"\n")
        f.write("geographical_region = \"geographical_region.txt\"\n")
        f.write("seasons = \"seasons.txt\"\n")
        f.write("\n#############\n# SEA ZONES\n#############\n\n")
        # Note this groups all of the sea zones together; maybe we should split them up a bit.
        if len(waste_list) > 0:
            first_sea = pid_from_hex[waste_list[-1][-1]] + 1
        else:
            raise NotImplementedError
        if len(major_rivers) > 0:
            last_sea = major_rivers[0] - 1
        else:
            last_sea = last_pid - 1
        f.write(f"sea_zones = RANGE {{ {first_sea} {last_sea} }}")

        f.write("\n###############\n# MAJOR RIVERS\n###############\n\n")
        # TODO: Currently this doesn't try to lump them together at all, but it should.
        for major_river in major_rivers:
            f.write(f"river_provinces = LIST {{ {major_river} }}\n")
        
        f.write("\n########\n# LAKES\n########\n\n")
        # TODO: Add lakes. We currently don't generate any, but it might be a replacement for some wastelands.

        f.write("\n#####################\n# IMPASSABLE TERRAIN\n#####################\n\n")
        # Provinces that can be colored by ownership; basically all of the ones that we'll make.
        for waste in waste_list:
            # TODO Maybe check to see if it should be a range instead of a list if possible? Like this should be the same order.
            f.write("impassable_mountains = LIST { " + str(pid_from_hex[waste[0]]) + " }\n")

        f.write("\n############\n# WASTELAND\n############\n\n")
        # Provinces that can't be colored by ownership. Probably won't be used, except for seas.
        f.write("# IMPASSABLE SEA ZONES\n")
        f.write(f"impassable_seas = LIST {{ {last_pid} }}")  # TODO add inner sea if it's not 0


def make_definition(file_dir, rgb_from_pid, name_from_pid):
    with open(os.path.join(file_dir,"map_data", "definition.csv"),'w', encoding='utf8') as f:
        f.write('0;0;0;0;x;x;\n')
        for pid, rgb in rgb_from_pid.items():
            if pid in name_from_pid:
                name = name_from_pid[pid]
            else:
                name = "Unknown"
            f.write(';'.join([str(pid)] + [str(c) for c in rgb] + [name, 'x', '\n']))
        # f.write(f'{max(rgb_from_pid.keys()) + 1};255;255;255;x;x;\n') # I think I fixed the bug that needed this as a fix.


def copy_file(src_dir, title, area, file_name, out_dir):
    inf_loc = os.path.join(src_dir, title, area, file_name)
    if os.path.exists(inf_loc):
        with open(os.path.join(out_dir, area, file_name), 'w', encoding='utf8') as outf:
            with open(inf_loc, encoding='utf8') as inf:
                outf.write(inf.read())


def make_dynasties(empires, religions, src_dir, out_dir):
    '''Writes out the common/dynasties, history/characters, and history/titles files.'''
    os.makedirs(os.path.join(out_dir,"common", "dynasty_houses"), exist_ok=True)
    with open(os.path.join(out_dir,"common", "dynasty_houses", "00_dynasty_houses.txt"),'w', encoding='utf8') as outf:
        outf.write("")
    os.makedirs(os.path.join(out_dir,"history", "wars"), exist_ok=True)
    with open(os.path.join(out_dir,"history", "wars", "00_wars.txt"),'w', encoding='utf8') as outf:
        outf.write("")
    os.makedirs(os.path.join(out_dir,"history", "province_mappings"), exist_ok=True)
    with open(os.path.join(out_dir,"history", "province_mappings", "00_provinces.txt"),'w', encoding='utf8') as outf:
        outf.write("")
    os.makedirs(os.path.join(out_dir,"history", "characters"), exist_ok=True)
    os.makedirs(os.path.join(out_dir,"history", "titles"), exist_ok=True)
    os.makedirs(os.path.join(out_dir,"common", "coat_of_arms", "coat_of_arms"), exist_ok=True)
    os.makedirs(os.path.join(out_dir,"common", "dynasties"), exist_ok=True)
    areas = [os.path.join(*area_list) for area_list in [
        ["history", "characters"], 
        ["history", "titles"], 
        ["common", "coat_of_arms", "coat_of_arms"],
    ]]
    # Perhaps we should also replace the big 00_dynasties.txt? Or reconfigure coat_of_arms to do the same thing.
    with open(os.path.join(out_dir,"common", "dynasties", "00_dynasties.txt"),'w', encoding='utf8') as doutf:
        for religion in religions:
            for area in areas:
                copy_file(src_dir, f"r_{religion}", area, f"r_{religion}.txt", out_dir)
            inf_loc = os.path.join(src_dir, f"r_{religion}", "common", "dynasties", "00_dynasties.txt")
            if os.path.exists(inf_loc):
                with open(inf_loc, encoding='utf8') as inf:
                    doutf.write(inf.read())
        for titles in empires.values():
            for title in titles:
                for area in areas:
                    copy_file(src_dir, title, area, f"{title}.txt", out_dir)
                inf_loc = os.path.join(src_dir, title, "common", "dynasties", "00_dynasties.txt")
                if os.path.exists(inf_loc):
                    with open(inf_loc, encoding='utf8') as inf:
                        doutf.write(inf.read())

def make_innovations(src_dir, out_dir):
    '''Writes out the common/culture/innovations files.'''
    area = os.path.join("common", "culture", "innovations")
    os.makedirs(os.path.join(out_dir, area), exist_ok=True)
    for era in os.listdir(os.path.join(src_dir, "start", area)):
        with open(os.path.join(out_dir, area, era), 'w', encoding='utf8') as outf:
            with open(os.path.join(src_dir, "start", area, era), encoding='utf8') as inf:
                outf.write(inf.read())

def make_religions(empires, hs_county, src_dir, out_dir):
    '''Gather the relevant data and then write out the religion files.'''
    # First we get all of the holy sites from all of the titles.
    hs_dict = {}
    used_hs = set()
    religions = set()
    os.makedirs(os.path.join(out_dir, "common", "religion", "holy_sites"), exist_ok=True)
    with open(os.path.join(out_dir, "common", "religion", "holy_sites", "00_holy_sites.txt"), 'w', encoding='utf8') as outf:
        for titles in empires.values():
            for title in titles:
                hs_path = os.path.join(src_dir, title, "common", "religion", "holy_sites")
                if os.path.exists(hs_path):
                    for religion_file in os.listdir(hs_path):
                        hs_names = []
                        with open(os.path.join(hs_path, religion_file), encoding='utf8') as inf:
                            for line in inf.readlines():
                                if line[0] not in ['\t', '}', '#'] and '=' in line:
                                    if line[0] == '\ufeff':
                                        line = line[1:]
                                    hs_names.append(line.split('=')[0].strip())
                                    used_hs.add(hs_names[-1])
                                outf.write(line)
                            outf.write("\n")
                        religion = religion_file.split('.')[0]
                        religions.add(RELIGION_DICT[religion])
                        if religion in hs_dict:
                            hs_dict[religion].extend(hs_names)
                        else:
                            hs_dict[religion] = hs_names
        # Now we write out the unused holy sites, which we blank out so that we can use the vanilla religions.
        for holy_site in HOLY_SITE_LIST:
            if holy_site not in used_hs:
                outf.write(f"{holy_site} = {{\n\tcounty = {hs_county}\n}}\n")
    # Now we know which religion files to write out, and what holy sites to give them.
    os.makedirs(os.path.join(out_dir, "common", "religion", "religions"), exist_ok=True)
    for religion in religions:
        with open(os.path.join(out_dir, "common", "religion", "religions", f"00_{religion}.txt"), 'w', encoding='utf8') as outf:
            with open(os.path.join(src_dir, f"r_{religion}", "common", "religion", "religions", f"00_{religion}.txt"), encoding='utf8') as inf:
                for line in inf.readlines():
                    if 'holy_site = ' in line:
                        hs_group = line.split('=')[1].strip()
                        for hs in hs_dict.get(hs_group, []):
                            outf.write(f"\t\t\tholy_site = {hs}\n")
                    else:
                        outf.write(line)
    return religions

def copy_over(src_dir, out_dir, so_far=[]):
    '''Copy over files that have been modified to drop inappropriate events / references / etc.'''
    for top_level in os.listdir(os.path.join(src_dir, *so_far)):
        if os.path.isdir(os.path.join(src_dir, *so_far, top_level)):
            copy_over(src_dir, out_dir, so_far=so_far + [top_level])
        else:
            this_dir = os.path.join(out_dir, *so_far)
            os.makedirs(this_dir, exist_ok=True)
            with open(os.path.join(this_dir, top_level), 'w', encoding='utf8') as outf:
                with open(os.path.join(src_dir, *so_far, top_level), encoding='utf8') as inf:
                    outf.write(inf.read())




if __name__ == '__main__':
    #Eventually we're going to use a config file to run most of this.
    make()
