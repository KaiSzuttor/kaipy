# This file is part of kaipy.
# Copyright (C) 2015  Kai Szuttor
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np


def second_legendre(pos1, pos2, direction):
    """
    Calculates second legendre polonomial of two given position vectors
    for the angle of (pos2-pos1, direction). direction can either be:
      - x (1,0,0)
      - y (0,1,0)
      - z (0,0,1)
    """
    vec1 = pos1 - pos2
    try:
        if (direction == 'x'):
            vec2 = np.array([1, 0, 0])
        elif (direction == 'y'):
            vec2 = np.array([0, 1, 0])
        elif (direction == 'z'):
            vec2 = np.array([0, 0, 1])
    except:
        raise ValueError("Argument must be 'x','y' or 'z'")
    v1_norm = np.sqrt((vec1 * vec1).sum())
    v2_norm = np.sqrt((vec2 * vec2).sum())
    cos_angle = np.dot(vec1, vec2) / (v1_norm * v2_norm)
    second_legendre = 0.5 * (3 * cos_angle * cos_angle - 1)
    return second_legendre
