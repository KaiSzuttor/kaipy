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
cimport numpy as np

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

def second_legendre(np.ndarray[DTYPE_t, ndim=1] pos1,
		np.ndarray[DTYPE_t,	ndim=2] pos2, np.str direction):
    """ Calculate the second legendre polonomial.

    Calculates second legendre polonomial of two given position vectors
    for the angle of (pos2-pos1, direction). direction can either be:

    Parameters
    ----------
    pos1 : array_like
        First position vector.
    pos2 : array_like
        Second position vector.
    direction : str
        Direction ('x', 'y', or 'z')

    Returns
    -------
    float

    Raises
    ------
    ValueError
        If direction is not ('x', 'y' or 'z')

    """
    cdef np.ndarray[DTYPE_t, ndim=1] vec1 = pos1 - pos2
	cdef np.ndarray[DTYPE_t, ndim=1] vec2
    try:
        if (direction == 'x'):
            vec2 = np.array([1, 0, 0])
        elif (direction == 'y'):
            vec2 = np.array([0, 1, 0])
        elif (direction == 'z'):
            vec2 = np.array([0, 0, 1])
    except:
        raise ValueError("Argument must be 'x','y' or 'z'")
    cdef DTYPE v1_norm = np.sqrt((vec1 * vec1).sum())
    cdef DTYPE v2_norm = np.sqrt((vec2 * vec2).sum())
    cdef DTYPE cos_angle = np.dot(vec1, vec2) / (v1_norm * v2_norm)
    cdef DTYPE second_legendre = 0.5 * (3 * cos_angle * cos_angle - 1)
    return second_legendre


def rg2(np.ndarray[DTYPE_t, ndim=1] x):
    """ Square radius of gyration.

    Calculates the squared radius of gyration for coordinates x of a polymer.

    Parameters
    ----------
    x : array_like
        Array of length number of beads of the polymer.

    Returns
    -------
    float

    """
    cdef DTYPE_t rg2 = 0.0
    cdef np.ndarray[D_TYPE_t, ndim=1] r_mean = np.zeros(3)
    r_mean[:] = np.mean(x, axis=0)
    for i in range(len(x)):
        rg2 += np.dot((x[i] - r_mean), (x[i] - r_mean))
    rg2 /= len(x)
    return rg2


def rg2_compwise(np.ndarray[DTYPE_t, ndim=1] x):
    """ Square radius of gyration component-wise.

    Calculates the componentwise squared radius of gyration
    for coordinates x of a polymer.

    Parameters
    ----------
    x: array_like
       Array of length number of beads of polymer.

    Returns
    -------
    float, float, float
    """
    cdef np.ndarray[DTYPE_t, ndim=1] rg2 = np.zeros(3)
    cdef np.ndarray[DTYPE_t, ndim=1] r_mean = np.zeros(3)
    r_mean[:] = np.mean(x, axis=0)
    for i in range(len(x)):
        rg2 += (x[i]-r_mean)**2
    rg2 /= len(x)
    return rg2[0], rg2[1], rg2[2]


def end_to_end_distance(np.ndarray[DTYPE_t, ndim=1] x):
    """ End to end distance of polymer.

    Calculates the absolute value of the end to end vector
    for coordinates x of a polymer.

    Parameters
    ----------
    x: array_like
        Array of length number of beads of polymer.

    Returns
    -------
    float
    """
    return np.linalg.norm(x[-1]-x[0])


def center_of_mass(np.ndarray[DTYPE_t, ndim=1] x,
		                  np.ndarray[DTYPE_t, ndim=1] mass=None):
    """ Center of mass of polymer.

    Calculates the center of mass of a polymer with
    given coordinates x of the monomers and optional
    array of mass values mass.

    Parameters
    ----------
    x: array_like
        Array of length number of beads of polymer.
    mass: Optional[array_like]
        Array of length number of beads of polymer.

    Returns
    -------
    array_like
    """
	cdef np.ndarray[DTYPE_t, ndim=1] com
    if (mass != None):
        com = np.sum([x[i,:]*mass[i] for i in range(len(mass))], axis=0)/np.sum(mass)
    else:
	com = np.sum(x, axis=0)/len(x)
    return com
