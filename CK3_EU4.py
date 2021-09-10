from ck3map import Terrain
from eu4map import EU4Terrain

TERRAIN_CONVERSION = {
    Terrain.farmlands: EU4Terrain.farmlands,
    Terrain.plains: EU4Terrain.grasslands,
    Terrain.floodplains: EU4Terrain.farmlands,
    Terrain.taiga: EU4Terrain.glacial,
    Terrain.wetlands: EU4Terrain.marsh,
    Terrain.steppe: EU4Terrain.steppe,
    Terrain.drylands: EU4Terrain.drylands,
    Terrain.oasis: EU4Terrain.desert,
    Terrain.desert: EU4Terrain.desert,
    Terrain.jungle: EU4Terrain.jungle,
    Terrain.forest: EU4Terrain.forest,
    Terrain.hills: EU4Terrain.hills,
    Terrain.mountains: EU4Terrain.mountains,
    Terrain.desert_mountains: EU4Terrain.mountains,
}