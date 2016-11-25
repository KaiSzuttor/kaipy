#!/usr/bin/env python

"""
Unit-test module for the kaipy.util module.
"""

import os
import unittest
import numpy as np
import h5py
from kaipy.util import h5md_pos

pos_unfolded = np.array([
    [[11.11, 1.21, 1.31],
     [2.11, 2.21, 12.31],
     [3.11, 3.21, 3.31],
     [4.11, 14.21, 4.31],
     [15.11, 15.21, 15.31]],
    [[-128.88, 11.22, 11.32],
     [22.12, 32.22, 42.32],
     [3.12, 3.22, 3.32],
     [-5.88, -5.78, -5.68],
     [15.12, 15.22, 5.32]],
    [[1.13, 1.23, 1.33],
     [22.13, 32.23, 22.33],
     [-6.87, 13.23, -6.67],
     [14.13, -5.77, 14.33],
     [5.13, -94.77, 5.33]],
    [[71.14, 341.24, 1.34],
     [22.14, 32.24, -17.66],
     [3.14, -96.76, 3.34],
     [14.14, -5.76, 14.34],
     [-4.86, 15.24, -4.66]]
    ])

pos_folded = np.array([
    [[1.11, 1.21, 1.31],
     [2.11, 2.21, 2.31],
     [3.11, 3.21, 3.31],
     [4.11, 4.21, 4.31],
     [5.11, 5.21, 5.31]],
    [[1.12, 1.22, 1.32],
     [2.12, 2.22, 2.32],
     [3.12, 3.22, 3.32],
     [4.12, 4.22, 4.32],
     [5.12, 5.22, 5.32]],
    [[1.13, 1.23, 1.33],
     [2.13, 2.23, 2.33],
     [3.13, 3.23, 3.33],
     [4.13, 4.23, 4.33],
     [5.13, 5.23, 5.33]],
    [[1.14, 1.24, 1.34],
     [2.14, 2.24, 2.34],
     [3.14, 3.24, 3.34],
     [4.14, 4.24, 4.34],
     [5.14, 5.24, 5.34]]
    ])

class H5mdPos(unittest.TestCase):
    """
    Test the h5md_pos method.
    """

    h5_fh = None

    @classmethod
    def setUpClass(cls):
        """
        Prepare a .h5 file as reference.
        """
        h5_fh = h5py.File('test.h5', 'w')

        # h5 groups
        p_group = h5_fh.create_group("/particles/atoms/position")
        id_group = h5_fh.create_group("/particles/atoms/id")
        im_group = h5_fh.create_group("/particles/atoms/image")
        box_group = h5_fh.create_group("particles/atoms/box")

        # h5 datasets
        pos_ds = p_group.create_dataset(
            "value", (4, 5, 3), maxshape=(None, None, 3))
        id_ds = id_group.create_dataset(
            "value", (4, 5, 1), maxshape=(None, None, 1))
        im_ds = im_group.create_dataset(
            "value", (4, 5, 3), maxshape=(None, None, 3))
        box_ds = box_group.create_dataset("edges", (3,), maxshape=(3,))

        # hardcoded test_array, pattern "particle_id.x/y/z.timestep"
        test_pos = np.array([
            [[3.11, 3.21, 3.31],
             [1.11, 1.21, 1.31],
             [4.11, 4.21, 4.31],
             [2.11, 2.21, 2.31],
             [5.11, 5.21, 5.31]],
            [[3.12, 3.22, 3.32],
             [4.12, 4.22, 4.32],
             [1.12, 1.22, 1.32],
             [2.12, 2.22, 2.32],
             [5.12, 5.22, 5.32]],
            [[3.13, 3.23, 3.33],
             [1.13, 1.23, 1.33],
             [2.13, 2.23, 2.33],
             [4.13, 4.23, 4.33],
             [5.13, 5.23, 5.33]],
            [[5.14, 5.24, 5.34],
             [1.14, 1.24, 1.34],
             [2.14, 2.24, 2.34],
             [4.14, 4.24, 4.34],
             [3.14, 3.24, 3.34]]
        ])

        test_id = np.array([
            [[3], [1], [4], [2], [5]],
            [[3], [4], [1], [2], [5]],
            [[3], [1], [2], [4], [5]],
            [[5], [1], [2], [4], [3]]
        ])

        test_im = np.array([
            [[0, 0, 0],
             [1, 0, 0],
             [0, 1, 0],
             [0, 0, 1],
             [1, 1, 1]],
            [[0, 0, 0],
             [-1, -1, -1],
             [-13, 1, 1],
             [2, 3, 4],
             [1, 1, 0]],
            [[-1, 1, -1],
             [0, 0, 0],
             [2, 3, 2],
             [1, -1, 1],
             [0, -10, 0]],
            [[-1, 1, -1],
             [7, 34, 0],
             [2, 3, -2],
             [1, -1, 1],
             [0, -10, 0]]
        ])

        test_box = np.array(
            [10.0, 10.0, 10.0]
        )

        pos_ds[:, :, :] = test_pos
        id_ds[:, :, :] = test_id
        im_ds[:, :, :] = test_im
        box_ds[:] = test_box

        h5_fh.close()


        cls.h5_fh = h5py.File("test.h5", 'r')

    def test_folded_single(self):
        """
        Test the h5md_pos method for a single timestep and folded coordinates.
        """
        for i in range(pos_folded.shape[0]):
            self.assertTrue(np.allclose(h5md_pos(self.h5_fh,
                                                 i,
                                                 folded=True),
                                        pos_folded[i]))

    def test_unfolded_single(self):
        """
        Test the h5md_pos method for a single timestep and un-folded
        coordinates.
        """
        for i in range(pos_unfolded.shape[0]):
            self.assertTrue(np.allclose(h5md_pos(self.h5_fh,
                                                 i,
                                                 folded=False),
                                        pos_unfolded[i]))

    def test_folded_array(self):
        """
        Test the h5md_pos method for multiple timesteps and folded
        coordinates.
        """
        self.assertTrue(np.allclose(h5md_pos(self.h5_fh,
                                             np.arange(0, 4),
                                             folded=True),
                                    pos_folded))

    def test_unfolded_array(self):
        """
        Test the h5md_pos method for multiple timesteps and un-folded
        coordinates.
        """
        self.assertTrue(np.allclose(h5md_pos(self.h5_fh,
                                             np.arange(0, 4),
                                             folded=False),
                                    pos_unfolded))

    @classmethod
    def tearDownClass(cls):
        os.remove("test.h5")


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(H5mdPos)
    unittest.TextTestRunner(verbosity=2).run(suite)
