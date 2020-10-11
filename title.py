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
