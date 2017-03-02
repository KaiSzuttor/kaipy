#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module for MPI parallel calculations.
"""


from __future__ import print_function
import abc
import logging
import numpy as np

LOGGER = logging.getLogger(__name__)


class ParallelTrajectory(object):
    """
    Class that provides MPI parallelization for the calculation of given
    observable on a given trajectory.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, **kwargs):
        """
        Parameters:
        -----------
        comm : mpi4py.MPI.Intracomm
              MPI communicator.
        obs : function
              Function to be used for the calculation.
        res_shape : tuple
                    Shape of data returned by obs (ommit the time dimesnional contribution).
        n_ts : int
               Number of timesteps to consider.
        stride : int
                Timestep stride.
        offset : int
                 Timestep offset.


        """
        self.total_result = None
        self.comm = kwargs['comm']
        self.mpi_rank = self.comm.Get_rank()
        self.mpi_size = self.comm.Get_size()
        self.timestep_range = self.calc_range(self.mpi_rank, kwargs['n_ts'],
                                              kwargs['stride'], kwargs['offset'])

    def calc_range(self, rank, n_ts, stride, offset=0):
        """
        Calculate the timestep range for the current rank.

        Parameters:
        -----------
        n_ts : int
               Number of total timesteps.
        stride : int
                 Timestep stride.

        Returns:
        --------
        array_like
            Indices of the timesteps to calculate.

        """
        start = rank * (n_ts - offset) / self.mpi_size
        stop = (rank + 1) * (n_ts - offset) / self.mpi_size
        if rank == 0:
            return np.arange(0, stop, stride) + offset
        elif rank == self.mpi_size - 1:
            res = np.arange(start, n_ts, stride)
            i = 1
            while res[0] % stride != 0:
                res = np.arange(start + i, n_ts, stride)
                i += 1
            res += offset
            while res[-1] > n_ts:
                res = np.arange(res[0], res[-1] - stride, stride)
            return res
        else:
            res = np.arange(start, stop, stride)
            i = 1
            while res[0] % stride != 0:
                res = np.arange(start + i, stop, stride)
                i += 1
            return res + offset

    @abc.abstractmethod
    def run(self):
        """
        Method to perform the actual commputation.
        """
        pass

    @abc.abstractmethod
    def communicate(self):
        """
        Method to communicate results.
        """
        pass


class H5mdParallelTrajectory(ParallelTrajectory):
    """
    Parallel evaluation for H5MD files.
    """

    def __init__(self, **kwargs):
        """
        Parameters:
        -----------

        h5md_file : h5py._hl.files.File
                    Instance of a H5MD file object.
        """
        # pylint: disable=too-many-instance-attributes
        super(H5mdParallelTrajectory, self).__init__(**kwargs)
        self.h5md = {}
        self.h5md['file'] = kwargs['h5md_file']
        try:
            self.h5md['pos'] = self.h5md['file'][
                '/particles/atoms/position/value']
        except ValueError:
            raise ValueError(
                "H5MD file does not contain valid position dataset.")
        try:
            self.h5md['time'] = self.h5md['file'][
                '/particles/atoms/position/time']
        except ValueError:
            raise ValueError(
                "H5MD file does not contain valid position dataset.")
        self.obs = kwargs['obs']
        self.mpi_buffer = np.zeros(
            ((self.timestep_range.shape[0],) + kwargs['res_shape']))
        self.n_ts = kwargs['n_ts']
        self.stride = kwargs['stride']
        self.offset = kwargs['offset']
        self.res_shape = kwargs['res_shape']

    def run(self, *args):
        for j, i in enumerate(self.timestep_range):
            self.mpi_buffer[j] = self.obs(self.h5md['pos'][i, :, :], *args)
        logging.debug("Rank: {}, Start: {}, Stop: {}".format(self.mpi_rank,
                                                             self.timestep_range[
                                                                 0],
                                                             self.timestep_range[-1]))
        logging.debug("Rank: {}, mpi_buffer shape: {}".format(self.mpi_rank,
                                                              self.timestep_range.shape))

    def communicate(self):
        if self.mpi_rank == 0:
            self.total_result = self.mpi_buffer
            for j in range(1, self.mpi_size):
                node_range = self.calc_range(
                    j, self.n_ts, self.stride, self.offset)
                recv_buffer = np.zeros(
                    ((node_range.shape[0],) + self.res_shape))
                logging.debug(
                    "Shape of recv_buffer: {}.".format(recv_buffer.shape))
                self.comm.Recv(recv_buffer, source=j, tag=int(j))
                self.total_result = np.concatenate(
                    (self.total_result, recv_buffer), axis=0)
        else:
            logging.debug("Shape of send buffer: {}.".format(
                self.mpi_buffer.shape))
            self.comm.Send(self.mpi_buffer, dest=0, tag=int(self.mpi_rank))
