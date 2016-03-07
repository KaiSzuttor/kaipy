# This file is part of kaipy.
# Copyright (C) 2016  Kai Szuttor
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
    vec1 = pos1 - pos2
    try:
        if (direction == 'x'):
            vec2 = np.array([1, 0, 0])
        elif (direction == 'y'):
            vec2 = np.array([0, 1, 0])
        elif (direction == 'z'):
            vec2 = np.array([0, 0, 1])
        v1_norm = np.sqrt((vec1 * vec1).sum())
        v2_norm = np.sqrt((vec2 * vec2).sum())
        cos_angle = np.dot(vec1, vec2) / (v1_norm * v2_norm)
        second_legendre = 0.5 * (3 * cos_angle * cos_angle - 1)
        return second_legendre
    except:
        raise ValueError("Argument must be 'x','y' or 'z'")
    


def rg2(x):
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
    rg2 = 0.0
    r_mean = np.zeros(3)
    r_mean[:] = np.mean(x, axis=0)
    for i in range(len(x)):
        rg2 += np.dot((x[i] - r_mean), (x[i] - r_mean))
    rg2 /= len(x)
    return rg2


def rg2_compwise(x):
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
    rg2 = np.zeros(3)
    r_mean = np.zeros(3)
    r_mean[:] = np.mean(x, axis=0)
    for i in range(len(x)):
        rg2 += (x[i]-r_mean)**2
    rg2 /= len(x)
    return rg2[0], rg2[1], rg2[2]


def end_to_end_distance(x):
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


def center_of_mass(x, mass=None):
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
    if (mass is not None):
        com = np.average(x, axis=0, weights=mass)
    else:
        com = np.average(x, axis=0)
    return com
	    

def radial_distribution(h5md_pos, h5md_species, SPECIES_1, SPECIES_2, TIMESTEP_MIN, TIMESTEP_MAX, BOX_L, N_BINS, R_MIN, R_MAX=None):
    """ Radial distribution function.

    Calculates the radial distribution function
    (RDF) for a trajectory array h5md_pos with the 
    content [timesteps, particles, xyz] in the 
    h5md format (see http://nongnu.org/h5md/ 
    for details). 

    Parameters
    ----------
    h5md_pos: array_like
        Three dimensional array of the particle
        trajectory.
    h5md_species: array like
        One dimensional array of length number
        of particles (h5md_pos.shape[1]).
    SPECIES_1: int
        First species for which the RDF is 
        calculated.
    SPECIES_2: int
        Second species (see SPECIES_1).
    TIMESTEP_MIN: int
        Lower limit of the range of timesteps
        for the RDF calculation.
    TIMESTEP_MAX: int
        Upper limit of the range (see TIMESTEP_MIN).
    BOX_L: float
        Length of the cubic simulation box.
    N_BINS: int
        Number of bins for the RDF.
    R_MIN: float
        Minimum radial distance for the RDF.
    R_MAX: float
        Maximum radial distance for the RDF. Defaults to BOX_L/2.
    Returns
    -------
    array_like, array_like
        The first array is the RDF, the second histogram
        is the array of the midpositons of the bins.
    """
    if R_MAX is None:
        R_MAX = 0.5 * BOX_L # due to minimum image convention
    VOLUME = BOX_L*BOX_L*BOX_L
    step = (R_MAX-R_MIN)/float(N_BINS)
    bin_edges = np.linspace(R_MIN, R_MAX, num=N_BINS+1, endpoint=True)
    N_SPECIES_1 = h5md_pos[0,h5md_species[:]==SPECIES_1,0].shape[0]
    N_SPECIES_2 = h5md_pos[0,h5md_species[:]==SPECIES_2,0].shape[0]
    hist_master = np.zeros(N_BINS)
    for i in range(TIMESTEP_MIN, TIMESTEP_MAX):
        bin_mids = (bin_edges + 0.5*step)[:-1]
        hist = np.zeros(N_BINS)
        count = 0 
        SPECIES_1_pos = h5md_pos[i,h5md_species[:]==SPECIES_1,:]
        SPECIES_2_pos = h5md_pos[i,h5md_species[:]==SPECIES_2,:]
        for j in range(N_SPECIES_1):
            for k in range(N_SPECIES_2):
                diff_vector = SPECIES_1_pos[j,:] - SPECIES_2_pos[k,:]
                diff_vector_minimum_image = diff_vector - BOX_L * np.rint(diff_vector/BOX_L)
                norm_diff_vector_minimum_image = np.linalg.norm(diff_vector_minimum_image)
                if (norm_diff_vector_minimum_image > R_MIN and norm_diff_vector_minimum_image < R_MAX):
                    hist_bin = np.digitize(np.array([norm_diff_vector_minimum_image,]), bin_edges)[0]-1
                    hist[hist_bin] += 1
                    count += 1

        for i in range(0,len(bin_edges)-1):
            r_in = bin_edges[i]
            r_out = r_in + step
            bin_volume = 4.0/3.0 * np.pi*((r_out*r_out*r_out) - (r_in*r_in*r_in))
            hist[i] *= VOLUME/ ( bin_volume*float(count) )
        hist_master += hist
    hist_master /= float(TIMESTEP_MAX-TIMESTEP_MIN)
    return hist_master, bin_mids

