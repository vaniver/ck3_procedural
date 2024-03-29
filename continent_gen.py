#!/usr/bin/env python

import pdb
import random
from itertools import combinations, product

import numpy as np

from doodle import Doodler
from tile import Tile
from cube import Cube
from chunk_split import split_chunk

CENTER_SIZE_LIST = [7,5,5,5]
KINGDOM_SIZE_LIST = [[6,4,4,4,4], [4,4,3,3], [4,4,3,3], [4,4,3]]
BORDER_SIZE_LIST = [4,4,4]


def k_col():
    return (np.random.randint(128, 256), 64, 64)


def d_col():
    return (64,np.random.randint(128, 256), np.random.randint(128, 256))


def c_col():
    return (np.random.randint(64, 256),64,np.random.randint(64, 256))


def random_rgb_tuple(size_list):
    if isinstance(size_list[0], int):
        return (d_col(),
                [c_col() for _ in size_list])
    else:
        return (k_col(),
                [random_rgb_tuple(sublist) for sublist in size_list])


def make_capital_county(c_size=5, origin=Cube(), coastal=True, rgb=None):
    '''Makes a county where all provinces neighbor the central province.'''
    rgb = rgb or c_col()
    a = Tile(origin=origin, rgb=rgb)
    cube_list = list(Cube(0,0,0).neighbors())
    if coastal and c_size<7:
        a.water_list.append(cube_list[c_size-1])
    cube_list = cube_list[:c_size-1]
    for el in cube_list:
        a.add_hex(el)
    return a


def make_capital_duchy(origin=Cube(0,0,0), size_list=KINGDOM_SIZE_LIST[0], rgb_tuple=None, coastal=True):
    '''Makes a duchy whose capital county is clumped.
    If coastal=True and size_list[0]<7, one water hex bordering the capital will be included.
    rgb_tuple should be ((r,g,b),[(r,g,b)*])'''
    rgb_tuple = rgb_tuple or random_rgb_tuple(size_list)
    capital_county = make_capital_county(c_size=size_list[0], coastal=coastal, rgb=rgb_tuple[1][0])
    duchy = Tile(origin=origin, tile_list=[capital_county],
                 hex_list=[], rgb=rgb_tuple[0])
    for idx, el in enumerate(size_list[1:]):
        try:
            duchy.add_new_tile(el, rgb = rgb_tuple[1][idx + 1])
        except:
            print(idx, rgb_tuple)
            raise ValueError
    if coastal:
        drhl = duchy.real_hex_list()
        if check_water_access(drhl, duchy.real_water_list(), max([el.mag() for el in drhl])):
            return duchy
        else:
            return make_capital_duchy(origin, size_list, rgb_tuple, coastal)
    else:
        return duchy


def make_original_center_duchy(origin=Cube(0,0,0), size_list=CENTER_SIZE_LIST, rgb_tuple=None):
    rgb_tuple = rgb_tuple or random_rgb_tuple(size_list)
    capital_county = make_capital_county(c_size=size_list[0], coastal=False, rgb=rgb_tuple[1][0])
    duchy = Tile(origin=origin, tile_list=[capital_county],
                 hex_list=[], rgb=rgb_tuple[0])
    for idx, c_size in enumerate(size_list[1:]):
        duchy.add_new_tile(c_size, rgb=rgb_tuple[1][idx + 1], capital=Cube(0,-2,2).rotate_right(idx*2))
    return duchy
        

def make_kingdom(origin=Cube(0,0,0), size_list = KINGDOM_SIZE_LIST, rgb_tuple=None, coastal=True,):
    """rgb_tuple is complicated. For each level, the left element is the rgb of the title for that tile,
    and the right element is a list of rgb_tuples for the tiles the next element below 
    (or a list of rgb tuples for baronies)."""
    rgb_tuple = rgb_tuple or random_rgb_tuple(size_list)
    kingdom = Tile(origin=origin, tile_list=[make_capital_duchy(size_list=size_list[0], coastal=coastal,
                   rgb_tuple=rgb_tuple[1][0])], hex_list=[], rgb=rgb_tuple[0])
    d_idx = 1
    while d_idx < len(size_list):
        duchy_size_list = size_list[d_idx]
        krhl = kingdom.relative_hex_list()
        krwl = kingdom.relative_water_list()
        new_county = Tile.new_tile(duchy_size_list[0], rgb=rgb_tuple[1][d_idx][1][0])
        if new_county.move_into_place([kingdom.relative_neighbors()], krhl, krwl):
            new_duchy = Tile(origin=Cube(0,0,0), tile_list = [new_county], hex_list=[], rgb=rgb_tuple[1][d_idx][0])
            for c_idx, county_size_list in enumerate(duchy_size_list[1:]):
                new_duchy.add_new_tile(county_size_list, cant=krhl + krwl, rgb=rgb_tuple[1][d_idx][1][c_idx])
            if check_water_access(krhl + new_duchy.relative_hex_list(), krwl):
                kingdom.add_tile(new_duchy)
                d_idx += 1
    return kingdom


def new_continent_gen(num_kingdoms=3):
    assert num_kingdoms in [3, 4, 5]
    center = make_original_center_duchy(size_list=CENTER_SIZE_LIST)
    kingdoms = [make_kingdom(size_list = KINGDOM_SIZE_LIST) for _ in range(num_kingdoms)]
    cen_nbrs = center.neighbors()
    k_r_bnds = [kingdom.relative_boundary() for kingdom in kingdoms]
    port_locs = [kingdom.relative_water_list()[0] for kingdom in kingdoms]
    unfinished = True
    num_outer_tries = 0
    errors = {}
    while unfinished:
        num_outer_tries += 1
        unfinished, continent = inner_continent_gen(center, kingdoms, cen_nbrs, k_r_bnds, port_locs)
        if unfinished:
            print(continent)
            if continent in errors:
                errors[continent] += 1
            else:
                errors[continent] = 1
    print(errors)
    return continent


def inner_continent_gen(center, kingdoms, cen_nbrs, k_r_bnds, port_locs):
    continent = Tile(hex_list=[])
    continent.add_tile(center)
    for k_idx in range(3):
        if not move_kingdom_into_place(continent, kingdoms[k_idx], cen_nbrs):
            return True, 'move into place'
    if not(check_water_access(continent.real_hex_list(), continent.real_water_list())):
        return True, 'water access 0'
    assigned = continent.real_total_list()
    c_dist = calculate_distances(center, assigned, 10)[0]
    k_dists = calculate_distances(kingdoms[:3], assigned, sum(BORDER_SIZE_LIST))
    for a_idx, b_idx in combinations(range(3), r=2):
        allowable = [el for el in k_dists[a_idx] if el in k_dists[b_idx]]
        ranking = {el: c_dist[el] if el in c_dist else el.mag() for el in allowable}
        new_duchies = divide_into_duchies(BORDER_SIZE_LIST, 2, get_chunks(allowable), k_dists[a_idx], k_dists[b_idx], ranking)
        if new_duchies:
            for duchy in new_duchies:
                continent.add_tile(duchy)
        else:
            return True, 'border duchies 3'
    if not(check_water_access(continent.real_hex_list(), continent.real_water_list())):
        return True, 'water access bd 3'
    print('Added 3 kingdoms!')
    if len(kingdoms) > 3:
        if not inner_add_triangle(continent, kingdoms, 3):
            return True, 'add fourth kingdom'
        print('Added 4 kingdoms!')
    if len(kingdoms) > 4:
        if not inner_add_triangle(continent, kingdoms, 4):
            return True, 'add fifth kingdom'
        print('Added 5 kingdoms!')
    return False, continent  #(continent, kingdoms)


def sort_hexlist(list_to_sort, ranking):
    return [pair[1] for pair in sorted([(ranking[el], el) for el in list_to_sort])]


def add_center_duchy(size_list, allowable_chunks, a_dist, b_dist, ranking):
    '''Given a list of necessary sizes (size_list), and a list of list of hexes (allowable_chunks), 
    attempt to create a center duchy, where counties are adjacent to both a and b (has a hex with 1 a_dist and 1 b_dist).
    Ranking is a dictionary of all (base) elements in allowable_chunks.
    Returns False if it doesn't find a solution in time.'''
    size = sum(size_list)
    while len(allowable_chunks) > 0:
        chunk = allowable_chunks.pop(0)
        if len(chunk) >= size:
            poss_centers = sort_hexlist([el for el in chunk if all([nel in chunk for nel in el.neighbors()])], ranking)
            a_adj = sort_hexlist([el for el in chunk if a_dist[el] == 1], ranking)
            b_adj = sort_hexlist([el for el in chunk if b_dist[el] == 1], ranking)
            for center in poss_centers:
                duchy = Tile(hex_list=[], tile_list=[make_capital_county(size_list[0], origin=center,coastal=False)], rgb=d_col())
                c_nbrs = [el for el in duchy.tile_list[0].neighbors() if el in chunk]
                drhl = duchy.real_hex_list()
                a_county = add_center_county(size_list[1], c_nbrs, a_adj, [el for el in chunk if el not in drhl])
                if a_county:
                    duchy.add_tile(a_county)
                else:
                    continue
                drhl = duchy.real_hex_list()
                c_nbrs = [el for el in duchy.tile_list[0].neighbors() if el in chunk and el not in drhl]
                b_county = add_center_county(size_list[2], duchy.tile_list[0].neighbors(), b_adj, chunk)
                if b_county:
                    duchy.add_tile(b_county)
                else:
                    continue
                for _ in range(20):
                    try:
                        duchy.add_bordering_tile(size_list[3], rgb=c_col(), only=chunk, ranking=ranking)
                        break
                    except:
                        continue
                return duchy
    return False


def add_center_county(size, c_nbrs, adj, chunk):
    for outer_try_count in range(30):
        candidate = set()
        candidate.add(random.sample(c_nbrs, k=1)[0])
        candidate.add(random.sample(adj, k=1)[0])
        inner_try_count = 0
        while len(candidate) < size and inner_try_count < 30:
            pick = random.sample(candidate, k=1)[0]
            opts = [el for el in pick.neighbors() if el in chunk]
            if len(opts) >= 1:
                candidate.add(random.choice(opts))
        candidate = list(candidate)
        if len(get_chunks(candidate)) == 1:
            return Tile(hex_list=candidate, rgb=c_col())
    return False


def inner_add_triangle(continent, kingdoms, c_idx):
    assigned = continent.real_total_list()
    k_dists = calculate_distances(kingdoms[:c_idx], assigned, sum(CENTER_SIZE_LIST) - 4)
    border_tile = Tile(tile_list=[tel for tel in continent.tile_list if tel.size < sum([sum(sublist) for sublist in KINGDOM_SIZE_LIST])])
    border_dists = calculate_distances(border_tile, assigned, sum(CENTER_SIZE_LIST) - 4)[0]
    for a_idx, b_idx in combinations(range(c_idx), r=2):
        allowable = [el for el in k_dists[a_idx] if el in k_dists[b_idx]]
        # We shouldn't bother to try to place the center in a spot that doesn't have ocean access.
        center_chunks = []
        center_allowable = []
        a_adj = []
        b_adj = []
        for chunk in get_chunks(allowable):
            this_a_adj = [el for el in chunk if k_dists[a_idx][el] == 1]
            this_b_adj = [el for el in chunk if k_dists[b_idx][el] == 1]
            if (len(this_a_adj) > 0) and (len(this_b_adj) > 0) and check_water_access(assigned, [chunk[0]]):
                center_chunks.append(chunk)
                center_allowable.extend(chunk)
                a_adj.extend(this_a_adj)
                b_adj.extend(this_b_adj)
        if len(a_adj) == 0:
            continue
        ranking = {el: border_dists.get(el, el.mag()) for el in allowable}
        need_center = True
        num_center_tries = 0
        while num_center_tries < 20 and need_center:
            num_center_tries += 1
            new_center = add_center_duchy(CENTER_SIZE_LIST, center_chunks, k_dists[a_idx], k_dists[b_idx], ranking)
            if new_center:
                need_center = not check_water_access(assigned + new_center.real_hex_list(), continent.real_water_list())
            if not need_center:
                #Add the new center
                continent.add_tile(new_center)
                #Add the new kingdom
                cont_real = continent.real_hex_list()
                cen_nbrs = [el for el in continent.tile_list[-1].neighbors() if el not in cont_real]
                for kingdom_tries in range(15):
                    if move_kingdom_into_place(continent, kingdoms[c_idx], cen_nbrs):
                        temp_assigned = continent.real_total_list()
                        temp_k_dists = calculate_distances(kingdoms[:c_idx + 1], temp_assigned, sum(BORDER_SIZE_LIST))
                        ranking = calculate_distances(continent.tile_list[-2], temp_assigned, 8)[0] #Maybe this should just be mag?
                        #Add the 4 new border duchies.
                        allowable = [el for el in temp_k_dists[a_idx] if el in temp_k_dists[c_idx]]
                        a_ranking = dict([(el, ranking[el]) if el in ranking else (el, temp_k_dists[a_idx][el] + temp_k_dists[c_idx][el]) for el in allowable])
                        new_duchies = divide_into_duchies(BORDER_SIZE_LIST, 2, get_chunks(allowable), temp_k_dists[a_idx], temp_k_dists[c_idx], a_ranking)
                        if new_duchies:
                            continent.add_tile(new_duchies[0])
                            continent.add_tile(new_duchies[1])
                            allowable = [el for el in temp_k_dists[b_idx] if el in temp_k_dists[c_idx]]
                            b_ranking = dict([(el, ranking[el]) if el in ranking else (el, temp_k_dists[b_idx][el] + temp_k_dists[c_idx][el]) for el in allowable])
                            new_duchies = divide_into_duchies(BORDER_SIZE_LIST, 2, get_chunks(allowable), temp_k_dists[b_idx], temp_k_dists[c_idx], b_ranking)
                            if new_duchies:
                                continent.add_tile(new_duchies[0])
                                continent.add_tile(new_duchies[1])
                                cont_real = continent.real_hex_list()
                                radius = max([el.mag() for el in cont_real])
                                if check_water_access(cont_real, continent.real_water_list(), radius):
                                    return True
                                else:
                                    for _ in range(4):
                                        continent.tile_list.pop()
                            else:
                                for _ in range(2):
                                    continent.tile_list.pop()
                        continent.tile_list.pop() #Remove the kingdom that we added that we couldn't build duchies to.
                    else:
                        num_next_tries = 10 #If we fail at adding the kingdom, then we shouldn't try again.
                continent.tile_list.pop() #Remove the central duchy that we added but couldn't find a good kingdom for.
    return False

def make_island_kingdom(water_height, origin=None, size_list = KINGDOM_SIZE_LIST, banned = [], weighted=True, min_mag=6, min_capital_coast=3, min_coast=2, max_tries = 1000, strait_prob = 0.5, center_bias = 0.5, coast_bias = 0.125):
    '''Given a dictionary from cubes to distance from shore, return a tile with duchies whose size are from size_list,
    and which are connected either directly or by straits (with probability strait_prob), and doesn't have any hexes in banned.
    The probability that a hex is selected as the origin is proportional to np.exp(-el.mag() * center_bias) * np.exp(water_height[el] * coast_bias),
    so high values of center_bias will make it closer to the center and high values of coast_bias will make it further from the shore.
    Tries max_tries times and returns False if it fails.'''
    assert min_capital_coast >= 3
    assert min_coast >= 2
    for _ in range(max_tries):
        island = Tile(hex_list = [], tile_list=[make_capital_duchy(d_size=size_list[0])])
        if origin:
            island.tile_list[0].origin = origin
        else:
            opts = [k for k, v in water_height.items() if v >= min_capital_coast and k.mag() >= min_mag] #center-coast-water-land means center has to be at least 3.
            probs = [np.exp(-el.mag() * center_bias) * np.exp(water_height[el] * coast_bias) for el in opts]
            probs /= sum(probs)
            island.tile_list[0].origin = np.random.choice(opts, p=probs)    
        allowable = [k for k, v in water_height.items() if v >= min_coast and k not in banned]
        if any([el not in allowable for el in island.real_hex_list()]):
            break
        for d_size_list in size_list[1:]:
            allocated = island.real_total_list()
            land_nbrs = island.neighbors()
            allowable = [el for el in allowable if el not in allocated]
            if np.random.rand() <= strait_prob:
                opts = [el for el in island.strait_neighbors() if el in allowable]
                new_origin = random.choice(opts)
                water_hexes = set()
                new_origin_nbrs = new_origin.neighbors()
                land = island.real_hex_list()
                for strait_neighbor in new_origin.strait_neighbors():
                    if strait_neighbor in land:
                        water_hexes.update([el for el in strait_neighbor.neighbors() if el in new_origin_nbrs])
                new_tile = Tile(hex_list=[new_origin], water_list = list(water_hexes))
                while len(new_tile.hex_list) < size:
                    new_neighbors = new_tile.relative_neighbors(weighted)
                    new_neighbors = [x for x in new_neighbors if x not in land_nbrs and x in allowable]
                    if len(new_neighbors) > 0:
                        new_tile.add_hex(random.choice(new_neighbors))
                    else:
                        break
                if len(new_tile.hex_list) == size:
                    island.add_tile(new_tile)
            else:
                opts = [el for el in island.neighbors() if el in allowable]
                new_origin = random.choice(opts)
                new_tile = Tile(hex_list=[new_origin])
                while len(new_tile.hex_list) < size:
                    new_neighbors = new_tile.relative_neighbors(weighted)
                    new_neighbors = [x for x in new_neighbors if x not in allocated and x in allowable]
                    if len(new_neighbors) > 0:
                        new_tile.add_hex(random.choice(new_neighbors))
                    else:
                        break
                if len(new_tile.hex_list) == size:
                    island.add_tile(new_tile)
        if len(island.real_hex_list()) == sum(size_list) and check_water_access(island.real_hex_list(), island.real_water_list(), max([el.mag() for el in island.real_hex_list()])):
            return island
    return False


def check_water_access(land, water, max_size=None, valid_dir=None, ocean_access = []):
    '''Ensure that all water has access to the ocean. Uses valid_dir to define boundaries and cut off the search early.
    If ocean_access is passed, it will trust that all elements of that list have access to the ocean.
    Returns computed ocean_access if successful for all water elements, and False if any water element is trapped.'''
    if len(water) == 0: #Suppose we call this check before we have any water sources we need to check on; then we need to return True, and behave correctly when we get that back next iteration.
        return True
    max_size = max_size or max([el.mag() for el in land])
    set_ocean_access = set(ocean_access)
    for source in water:
        invalid = True
        to_search = [el for el in source.neighbors() if not el in land ]
        searched = []
        while len(to_search)>0 and invalid:
            curr = to_search.pop() #Depth-first seems like the right call.
            if curr.mag() >= max_size: #We hit the outer edge.
                invalid = False
            elif valid_dir and not curr.valid(valid_dir): #We hit the med.
                invalid = False
            elif curr in ocean_access: #We hit something that's already got access.
                invalid = False
            else:
                to_search.extend([el for el in curr.neighbors() if not (el in land) and not (el in searched) and not (el in to_search)])
            searched.append(curr)
        if invalid:
            return False
        else:
            set_ocean_access.update(searched + to_search)
    return list(set_ocean_access)


def get_chunks(hex_list):
    ''' Figure out contiguous subsets, return as a list of list of hexes. '''
    N = len(hex_list)
    connection = np.eye(N)
    for i in range(N):
        for j in range(i):
            if hex_list[i].sub(hex_list[j]).mag() <= 1:
                connection[i, j] = connection[j, i] = 1

    chunks = []  # list of sets of indices
    never_visited = set(range(N))
    while never_visited:
        chunk = set()
        to_visit = {never_visited.pop()}
        while to_visit:
            next_visit = to_visit.pop()
            chunk.add(next_visit)
            # list of indices connected to next_visit that aren't in chunk
            newly_available = [i for i in range(N) if connection[next_visit, i] and i not in chunk]
            to_visit.update(newly_available)
            chunk.update(newly_available)
        never_visited.difference_update(chunk)
        chunks.append([hex_list[el] for el in chunk])
    return chunks


def check_contig(split, nbr_dict):
    """Checks that each of the lists of hexs in split is contiguous."""
    for group in split:
        never_visited = set(group)
        to_visit = set()
        if len(to_visit) == 0:
            continue
        to_visit.add(never_visited.pop())
        while len(to_visit) > 0:
            this_one = to_visit.pop()
            for el in nbr_dict[this_one]:
                if el in never_visited:
                    never_visited.remove(el)
                    to_visit.add(el)
        if len(never_visited) > 0:
            return False
    return True

def divide_into_counties(tile, size_list):
    '''Given a tile (assumed to just have hexes in its hex_list) and a size_list, return a tile
    with contiguous counties with sizes described by size_list.'''
    hex_list = [el for el in tile.hex_list]
    assert len(hex_list) == sum(size_list)
    counties = [[] for _ in size_list]
    num_left = [el for el in size_list]
    nbr_dict = {}
    for el in hex_list:
        nbr_dict[el] = [nel for nel in el.neighbors() if nel in hex_list]
    nn1 = [el for el in hex_list if len(nbr_dict[el]) == 1]
    idx = 0
    while len(nn1) > 0:
        single = nn1.pop()
        hex_list.remove(single)
        counties[idx].append(single)
        num_left[idx] -= 2
        nbr = nbr_dict[single][0]
        counties[idx].append(nbr)
        hex_list.remove(nbr)
        while len(nbr_dict[nbr]) == 2 and num_left[idx] > 0:
            nbr = [el for el in nbr_dict[nbr] if el not in counties[idx]][0]
            counties[idx].append(nbr)
            hex_list.remove(nbr)
            num_left[idx] -= 1
        if len(nbr_dict[nbr]) == 2 and num_left[idx] == 0:
            nn1.append([el for el in nbr_dict[nbr] if el not in counties[idx]][0])
        idx += 1

    possible_tiles = []
    while len(allowable_chunks) > 0:
        chunk = allowable_chunks.pop(0)
        if len(chunk) >= size:
            a_adj = [el for el in chunk if a_dist[el] == 1]
            b_adj = [el for el in chunk if b_dist[el] == 1]
            if len(a_adj) >= 1 and len(b_adj) >= 1:
                #It might be possible.
                candidate = set()
                sorted_chunk = [pair[1] for pair in sorted([(ranking.get(el, 999), el) for el in chunk])]
                candidate.add([el for el in sorted_chunk if el in a_adj][0])
                candidate.add([el for el in sorted_chunk if el in b_adj][0])
                others = [el for el in sorted_chunk if el not in candidate]
                while (len(candidate) < size) and len(others) > 0:
                    other = others.pop(0)
                    if any([other.sub(el).mag() == 1 for el in candidate]):
                        candidate.add(other)
                if len(candidate) == size:
                    candidate = list(candidate)
                    if len(get_chunks(candidate)) == 1:
                        possible_tiles.append(Tile(hex_list = candidate, rgb=d_col()))
                    if len(possible_tiles) == num_duchies:
                        return possible_tiles
                    if len(others) >= size:
                        remaining = [el for el in sorted_chunk if el not in candidate]
                        other_chunks = get_chunks(remaining)
                        for oc in other_chunks:
                            if len(oc) >= size:
                                allowable_chunks.insert(0, oc)
    #pdb.set_trace()
    return False


def duchy_from_snake(snake, size_list):
    '''Given a snake of baronies (where each should only border 1 or 2 neighbors), chop it up into counties of sizes determined by size_list.'''
    duchy = Tile(rgb=d_col(), hex_list=[])
    ind = 0
    for c_size in size_list:
        duchy.add_tile(Tile(rgb=c_col(), hex_list=snake[ind:ind+c_size]))
        ind += c_size
    #The capital is assumed to come first, but it should be in the middle.
    if len(size_list) > 2:
        duchy.tile_list.insert(0,duchy.tile_list.pop(1))
    return duchy


def duchy_from_chunk(chunk, size_list):
    '''Given a chunk of baronies (where each could potentially border many others), chop it up into contiguous counties with sizes determined by size_list.'''
    duchy = Tile(rgb=d_col(), hex_list=[])
    splits = split_chunk(chunk, size_list)
    for ind in range(len(size_list)):
        duchy.add_tile(Tile(rgb=c_col(), hex_list=splits[ind]))
    return duchy


def salvage_remainder(possible_tiles, new_tile, chunk, allowable_chunks, size):
    possible_tiles.append(new_tile)
    assigned = []
    for tile in possible_tiles:
        assigned.extend(tile.real_hex_list())
    remaining = [el for el in chunk if el not in assigned]
    if len(remaining) > size:
        other_chunks = get_chunks(remaining)
        for oc in other_chunks:
            if len(oc) >= size:
                allowable_chunks.insert(0, oc)
    return possible_tiles, allowable_chunks


def divide_into_duchies(size_list, num_duchies, allowable_chunks, a_dist, b_dist, ranking):
    '''Given a list of necessary sizes (size_list), and a list of list of hexes (allowable_chunks), 
    attempt to create num_duchies duchies with size hexes each, where each is adjacent to both a and b (has a hex with 1 a_dist and 1 b_dist).
    Ranking is a dictionary of all (base) elements in allowable_chunks.
    Returns False if it doesn't find a solution in time.'''
    assert num_duchies > 0
    size = sum(size_list)
    possible_tiles = []
    while len(allowable_chunks) > 0:
        chunk = allowable_chunks.pop(0)
        if len(chunk) < size:
            continue  # We can't make one, so don't bother trying.
        sorted_chunk = [pair[1] for pair in sorted([(ranking.get(el, el.mag()) + a_dist[el] + b_dist[el], el) for el in chunk])]
        if any([a_dist[el] == 1 for el in sorted_chunk]) and any([b_dist[el] == 1 for el in sorted_chunk]) and len(get_chunks(sorted_chunk[:size])) == 1:  # The closest size hexes are contiguous and border both.
            possible_tiles, allowable_chunks = salvage_remainder(possible_tiles, duchy_from_chunk(sorted_chunk[:size], size_list), chunk, allowable_chunks, size)
        else:
            sorted_chunk = [pair[1] for pair in sorted([(ranking.get(el, 999), el) for el in chunk])]
            a_adj = [el for el in sorted_chunk if a_dist[el] == 1]
            b_adj = [el for el in sorted_chunk if b_dist[el] == 1]
            if len(a_adj) == 0 or len(b_adj) == 0:
                continue  # We're not going to get adjacency to both.
            closest_a = a_adj[0]
            snake = [closest_a]
            disconnected = True
            while disconnected:
                closer_nbrs = [el for el in snake[-1].neighbors() if el in chunk and b_dist.get(el,999) < b_dist[snake[-1]]]
                if len(closer_nbrs) == 0:
                    break
                sorted_nbrs = [pair[1] for pair in sorted([(ranking.get(el, 999), el) for el in closer_nbrs])]
                snake.append(sorted_nbrs[0])
                if b_dist[snake[-1]] == 1:
                    disconnected = False
            if not disconnected and len(snake) >= size:  # I'm not sure why I thought this case was possible.
                overage = len(snake) - size
                if a_dist[snake[overage]] == 1:
                    snake = snake[overage:]
                    possible_tiles, allowable_chunks = salvage_remainder(possible_tiles, duchy_from_snake(snake, size_list), chunk, allowable_chunks, size)
            elif not disconnected:  # We have a valid snake, but too few.
                underage = size - len(snake)
                extendable = True
                while underage > 0 and extendable:
                    start_nbrs = [el for el in snake[0].neighbors() if el in chunk and ranking.get(el,999) <= ranking.get(snake[0]) and el not in snake]
                    if len(start_nbrs) > 0:
                        snake.insert(0,random.choice(start_nbrs))
                        underage -= 1
                    if underage > 0:
                        end_nbrs = [el for el in snake[-1].neighbors() if el in chunk and ranking.get(el,999) <= ranking.get(snake[-1]) and el not in snake]
                        if len(end_nbrs) > 0:
                            snake.append(random.choice(end_nbrs))
                            underage -= 1
                    if len(start_nbrs) == 0 and len(end_nbrs) == 0:
                        # Now we have to grow in the middle. 
                        extendable = False
                if underage == 0:
                    possible_tiles, allowable_chunks = salvage_remainder(possible_tiles, duchy_from_snake(snake, size_list), chunk, allowable_chunks, size)
                else:
                    duchy = Tile(rgb=d_col(), hex_list=[])
                    for c_size in size_list:
                        duchy.add_tile(Tile(rgb=c_col(), hex_list=[]))
                    assigned = []
                    ind = 0
                    while len(snake) > 0 and underage > 0 and ind < len(size_list):
                        el_nbrs = [el for el in snake[0].neighbors() if el in chunk and el not in snake and el not in assigned]
                        assigned.append(snake.pop(0))
                        duchy.tile_list[ind].hex_list.append(assigned[-1])
                        if len(duchy.tile_list[ind].hex_list) == size_list[ind]:
                            ind += 1
                            if ind == len(size_list):
                                break
                        if len(el_nbrs) > 0:
                            num_to_take = min(underage, size_list[ind] - len(duchy.tile_list[ind].hex_list))
                            added_now = random.sample(el_nbrs, min(num_to_take,len(el_nbrs)))
                            assigned.extend(added_now)
                            duchy.tile_list[ind].hex_list.extend(added_now)
                            if len(duchy.tile_list[ind].hex_list) == size_list[ind]:
                                ind += 1
                            # Note that we could keep going here, and check the neighbors of these neighbors, but I'm going to skip this for now.
                    if underage == 0:
                        if len(size_list) > 2:
                            duchy.tile_list.insert(0,duchy.tile_list.pop(1))
                        possible_tiles, allowable_chunks = salvage_remainder(possible_tiles, duchy, chunk, allowable_chunks, size)
        if len(possible_tiles) == num_duchies:
            return possible_tiles
    return False


def divide_into_duchies_old(size_list, num_duchies, allowable_chunks, a_dist, b_dist, ranking):
    '''Given a list of necessary sizes (size_list), and a list of list of hexes (allowable_chunks), 
    attempt to create num_duchies duchies with size hexes each, where each is adjacent to both a and b (has a hex with 1 a_dist and 1 b_dist).
    Ranking is a dictionary of all (base) elements in allowable_chunks.
    Returns False if it doesn't find a solution in time.'''
    assert num_duchies > 0
    size = sum(size_list)
    possible_tiles = []
    while len(allowable_chunks) > 0:
        chunk = allowable_chunks.pop(0)
        if len(chunk) >= size:
            a_adj = [el for el in chunk if a_dist[el] == 1]
            b_adj = [el for el in chunk if b_dist[el] == 1]
            if len(a_adj) >= 1 and len(b_adj) >= 1:
                #It might be possible.
                candidate = set()
                counties = [[] for _ in size_list]
                sorted_chunk = [pair[1] for pair in sorted([(ranking.get(el, 999), el) for el in chunk])]
                counties[1].append([el for el in sorted_chunk if el in a_adj][0])
                candidate.add(counties[1][0])
                counties[2].append([el for el in sorted_chunk if el in b_adj and el != counties[1][0]][0])
                candidate.add(counties[2][0])
                others = [el for el in sorted_chunk if el not in candidate]
                while (len(candidate) < size) and len(others) > 0:
                    other = others.pop(0)
                    adj = [any([other.sub(el).mag() == 1 for el in county]) for county in counties]
                    if any(adj):
                        candidate.add(other)
                        choice = 2
                        if adj[1] and adj[2]:
                            choice = np.argmin([999, size_list[1] - len(counties[1]), size_list[2] - len(counties[2])])
                        elif adj[1]:
                            choice = 1
                        elif adj[2]:
                            choice = 2
                        else:
                            choice = 0
                        if (choice == 1 and len(counties[1]) == size_list[1]) or (choice == 2 and len(counties[2]) == size_list[2]):
                            choice = 0
                        counties[choice].append(other)
                        others = [el for el in sorted_chunk if el not in candidate]
                if len(candidate) == size:
                    candidate = list(candidate)
                    if len(get_chunks(candidate)) == 1:
                        if all([len(get_chunks(county)) == 1 for county in counties]):
                            possible_tiles.append(Tile(hex_list = [], rgb=d_col(),
                                                       tile_list=[Tile(hex_list=county, rgb=c_col()) for county in counties]))
                    if len(possible_tiles) == num_duchies:
                        return possible_tiles
                    if len(others) >= size:
                        remaining = [el for el in sorted_chunk if el not in candidate]
                        other_chunks = get_chunks(remaining)
                        for oc in other_chunks:
                            if len(oc) >= size:
                                allowable_chunks.insert(0, oc)
    #pdb.set_trace()
    return False


def move_kingdom_into_place(stationary, mobile, targets, max_tries = 100, coastal = True):
    '''mobile is the kingdom to move; stationary is what to connect it to.
    targets are the acceptable places to add the kingdom.
    coastal is binary for whether the rotation should be port-sensitive.
    Movement is done in-place, but function returns True/False for success/failure.'''
    mobile.origin = Cube()
    mobile.rotation = 0
    mrbnd = mobile.relative_boundary()
    if coastal:
        port_loc = mobile.relative_water_list()[0]
    num_tries = 0
    while num_tries < max_tries and stationary.collision(mobile):
        k_nbr = random.sample(targets, k=1)[0]
        k_bnd = random.choice(mrbnd)
        rot = random.randint(0,5)
        if coastal:
            while (k_nbr.dot(k_bnd.sub(port_loc).rotate_right(rot)) > 0):
                rot = random.randint(0,5)
        mobile.rotation = rot
        k_bnd = k_bnd.rotate_right(rot)
        mobile.origin = k_nbr.sub(k_bnd)
        num_tries += 1
    if num_tries < max_tries:
        stationary.add_tile(mobile)
        return True
    else:
        return False


def calculate_distances(targets, assigned, max_dist):
    '''For each Tile in targets (or a single target Tile), calculate all the hexes that are within max_dist of them, 
    without going into any hexes in assigned.'''
    dists = []
    if isinstance(targets, Tile):
        targets = [targets]
    for target in targets:
        dist = {}
        assert isinstance(target, Tile)
        for el in target.neighbors():
            if el not in assigned:
                dist[el] = 1
        for distance in range(2, max_dist + 1):
            boundary = [el for el, d in dist.items() if d == distance - 1]
            for el in boundary:
                for el_nbr in el.neighbors():
                    if el_nbr not in assigned and el_nbr not in dist:
                        dist[el_nbr] = distance
        dists.append(dist)
    return dists


class BoundingHex:
    def __init__(self, tile, origin=Cube(), rotation=0, extra = 0):
        self.tile = tile
        self.origin = origin
        self.rotation = rotation
        self.min_x, self.min_y, self.min_z = 999, 999, 999
        self.max_x, self.max_y, self.max_z = -999, -999, -999
        for el in tile.real_hex_list():
            self.max_x = max(self.max_x, el.x)
            self.max_y = max(self.max_y, el.y)
            self.max_z = max(self.max_z, el.z)
            self.min_x = min(self.min_x, el.x)
            self.min_y = min(self.min_y, el.y)
            self.min_z = min(self.min_z, el.z)
        if extra > 0:
            self.max_x += extra
            self.max_y += extra
            self.max_z += extra
            self.min_x -= extra
            self.min_y -= extra
            self.min_z -= extra
        self.dists = self.distances()

    def __contains__(self, other):
        if isinstance(other, Cube):
            return (self.max_x >= other.x) and (self.min_x <= other.x) and \
                   (self.max_y >= other.y) and (self.min_y <= other.y) and \
                   (self.max_z >= other.z) and (self.min_z <= other.z)
        elif isinstance(other, Tile):
            return all([el in self for el in other.real_hex_list()])
        elif isinstance(other, list):
            return all([el in self for el in other])
        else:
            return False

    def corners(self, extra = 0):
        '''Returns the 6 corners of the BoundingHex, in adjacent clockwise order starting at the top. Use extra to increase the size of the bounding box.'''
        return [Cube(0-self.max_y-self.min_z, self.max_y + extra, self.min_z - extra), Cube(self.max_x + extra, 0-self.max_x-self.min_z, self.min_z - extra),
                Cube(self.max_x + extra, self.min_y - extra, 0-self.max_x-self.min_y), Cube(0-self.min_y-self.max_z, self.min_y - extra, self.max_z + extra),
                Cube(self.min_x - extra, 0-self.min_x-self.max_z, self.max_z + extra), Cube(self.min_x - extra, self.max_y + extra, 0-self.min_x-self.max_y)]

    def nearest_hex(self, other):
        '''Returns the nearest Cube inside the BoundingHex, or the supplied Cube if it's inside.'''
        # TODO
        pass
    
    def best_corner(self, angle):
        '''Returns the origin and rotation necessary to position this bounding_hex so that the highest-dist corner is nearest to the center, at the angle.'''
        corner_idx = np.argmax([self.dist(el) for el in self.corners()])
        rotation = (angle - corner_idx) % 6
        origin = self.corners(extra=1)[corner_idx].rotate_right(3+rotation)
        return (origin, rotation)

    def dist(self, other):
        adj_other = other.rotate_right(6-self.rotation).sub(self.origin)
        if adj_other in self.dists:
            return self.dists[adj_other]
        else:
            closest, dist_from = self.nearest_hex(adj_other)
            return self.dists[closest] + dist_from

    def distances(self):
        '''Constructs a dictionary that computes the distances to land for all Cubes in the BoundingHex.'''
        dists = {}
        boundary = set()
        land = self.tile.real_hex_list()
        distance = 0
        for el in land:
            dists[el] = distance
            boundary.update(el_nbr for el_nbr in el.neighbors() if el_nbr not in land and el_nbr in self)
        while (len(boundary) > 0):
            distance += 1
            next_boundary = set()
            for el in boundary:
                dists[el] = distance
                next_boundary.update(el_nbr for el_nbr in el.neighbors() if el_nbr not in dists and el_nbr not in boundary and el_nbr in self)
            boundary = next_boundary
        return dists


def make_world(cont_size_list = [3, 3, 3], island_size_list = [1, 1, 1], angles = [2, 4, 0], inner_sea = False):
    '''Create three continents, with number of continental kingdoms determined by cont_size_list, and arrange them around an inner sea.
    angles determines where the continents go; 2,4,0 is northwest, northeast, south; 3,5,1 is north, southeast, southwest.''' 
    # assert len(cont_size_list) == 3
    assert len(cont_size_list) == len(angles)
    assert len(cont_size_list) == len(island_size_list)
    world = Tile(hex_list=[])
    #Continents
    for cont_idx, cont_size in enumerate(cont_size_list):
        if cont_size == 0:
            world.add_tile(Tile(hex_list=[]))
            continue
        cont = new_continent_gen(num_kingdoms = cont_size)
        bounding_hex = BoundingHex(cont)
        cont.origin, cont.rotation = bounding_hex.best_corner(angles[cont_idx])
        world.add_tile(cont)
    #Check for straits
    # This will need to be fixed if we allow more than 3 continents.
    # TODO test more
    world.tile_list[-1].origin.add_in_place(Cube(0,1,-1))
    # connex = [0] * 3
    # while any([l == 0 for l in connex]):
    #     # Inch them closer one at a time until you can't get closer without touching.
    #     connex = [0] * 3
    #     for (ind_a, ind_b) in combinations(range(len(world.tile_list)), 2):
    #         tile_a = world.tile_list[ind_a]
    #         tile_b = world.tile_list[ind_b]
    #         strait_pairs, corner_pairs = tile_a.two_step_pairs(tile_b)
    #         if len(strait_pairs) + len(corner_pairs) >= 0:
    #             connex[ind_a] += 1
    #             connex[ind_b] += 1
    #     if 0 in connex:
    #         move_ind = connex.index(0)
    #         world.tile_list[move_ind].origin.add_in_place(Cube(0,1,-1).rotate_right(angles[move_ind]))
    #         world.tile_list[move_ind].rectify()
    # Inner sea
    if inner_sea:
        bounding_hex = BoundingHex(world)
        inner_sea = set()
        to_search = set([Cube()])
        while len(to_search) > 0:
            curr = to_search.pop()
            inner_sea.add(curr)
            to_search.update([el for el in curr.neighbors() if el in bounding_hex and bounding_hex.dist(el) >= 2 and el not in inner_sea])
        if len(inner_sea) > 22:
            #We have enough to make a duchy here
            water_height = {el: bounding_hex.dist(el) for el in inner_sea}
            inner_duchy = make_island_duchy(water_height)
            if inner_duchy:
                world.tile_list[-1].add_tile(inner_duchy)
                # inner_sea -= set(inner_duchy.inclusive_neighbors())
        #We should just drop some sicily-esque islands.
    #Outer islands
    for cont_idx, island_size in enumerate(island_size_list):
        for _ in range(island_size):
            land = world.real_hex_list()
            land_dist = calculate_distances(world, land, 3)[0]
            cont_dist = calculate_distances(world.tile_list[cont_idx], land, 6)[0]
            for k,v in cont_dist.items():
                if k in land_dist:
                    cont_dist[k] = min(v, land_dist[k])
            new_island = make_island_kingdom(cont_dist)
            if new_island:
                world.tile_list[cont_idx].add_tile(new_island)
            else:
                print("Failed to add an island!", cont_idx)
    return world


def make_island_duchy(water_height, size_list = CENTER_SIZE_LIST, rgb_tuple = None):
    '''Make a duchy with sizes according to size_list in the region defined by water_height.'''
    origin = max(water_height, key=water_height.get)
    rgb_tuple = rgb_tuple or random_rgb_tuple(size_list)
    capital_county = make_capital_county(c_size=size_list[0], coastal=False, rgb=rgb_tuple[1][0])
    duchy = Tile(origin=origin, tile_list=[capital_county],
                 hex_list=[], rgb=rgb_tuple[0])
    for idx, c_size in enumerate(size_list[1:]):
        duchy.add_bordering_tile(c_size, rgb=rgb_tuple[1][idx + 1], only=water_height)
    return duchy


if __name__ == '__main__':                
    world = make_world(cont_size_list=[5, 4, 3])