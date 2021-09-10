from cube import Cube
from dataclasses import dataclass
from enum import Enum

EU4Terrain = Enum('Terrain','farmlands grasslands coastland drylands coastal_desert savannah steppe desert open_sea marsh woods forest jungle hills highlands mountains glacial inland_sea')

TERRAIN_COLOR = {
    EU4Terrain.grasslands: 0,
    EU4Terrain.hills: 1,
    EU4Terrain.desert: 3,
    EU4Terrain.mountains: 6,
    EU4Terrain.marsh: 9,
    EU4Terrain.farmlands: 10,
    EU4Terrain.forest: 12,
    EU4Terrain.open_sea: 15,
    EU4Terrain.inland_sea: 17,
    EU4Terrain.coastal_desert: 18,
    EU4Terrain.savannah: 20,
    EU4Terrain.drylands: 23,
    EU4Terrain.jungle: 254,
}
TERRAIN_SNOW_COLOR = 16
TERRAIN_COAST_COLOR = 35

# HEIGHTMAP constants
WATER_HEIGHT = 94

class EU4Map:

    def __init__(self, max_x=5632, max_y=2048, hex_size=40, map_size=40, crisp=True, default=None):
        self.max_x = max_x
        self.max_y = max_y
        self.origin_x = max_x/2
        self.origin_y = max_y/2