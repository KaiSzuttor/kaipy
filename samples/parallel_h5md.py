#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy
import scipy.linalg
import h5py
from mpi4py import MPI
import matplotlib.pyplot as plt
from kaipy.parallel import H5mdParallelTrajectory

COMM = MPI.COMM_WORLD
H5MD_FILE = h5py.File('data.h5')


def end_to_end(x, polymer_length):
    return scipy.linalg.norm(x[polymer_length-1] - x[0])

ParaTrajectory = H5mdParallelTrajectory(comm=COMM, obs=end_to_end,
                                        res_shape=(1, ), n_ts=0, stride=2,
                                        offset=0, h5md_file=H5MD_FILE)
ParaTrajectory.run(50)
ParaTrajectory.communicate()
H5MD_FILE.close()

if COMM.Get_rank() == 0:
    plt.plot(ParaTrajectory.total_result)
    plt.show()
