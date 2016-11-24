#!/usr/bin/env python


import unittest
import numpy as np
import os
import h5py
from kaipy.util import h5md_pos
import reference_data


class H5md_Pos(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        h5_fh = h5py.File('test.h5', 'w')

        # h5 groups
        p_group = h5_fh.create_group("/particles/atoms/position")
        id_group = h5_fh.create_group("/particles/atoms/id")
        im_group = h5_fh.create_group("/particles/atoms/image")
        box_group = h5_fh.create_group("particles/atoms/box")

        # h5 datasets
        pos_ds = p_group.create_dataset("value", (4, 5, 3), maxshape=(None, None, 3))
        id_ds = id_group.create_dataset("value", (4, 5, 1), maxshape=(None, None, 1))
        im_ds = im_group.create_dataset("value", (4, 5, 3), maxshape=(None, None, 3))
        box_ds = box_group.create_dataset("edges",(3,), maxshape=(3,))

        # hardcoded test_array, pattern "particle_id.x/y/z.timestep"
        test_pos = np.array([
                [[3.11,3.21,3.31],
                 [1.11,1.21,1.31],
                 [4.11,4.21,4.31],
                 [2.11,2.21,2.31],
                 [5.11,5.21,5.31]],
                [[3.12,3.22,3.32],
                 [4.12,4.22,4.32],
                 [1.12,1.22,1.32],
                 [2.12,2.22,2.32],
                 [5.12,5.22,5.32]],
                [[3.13,3.23,3.33],
                 [1.13,1.23,1.33],
                 [2.13,2.23,2.33],
                 [4.13,4.23,4.33],
                 [5.13,5.23,5.33]],
                [[5.14,5.24,5.34],
                 [1.14,1.24,1.34],
                 [2.14,2.24,2.34],
                 [4.14,4.24,4.34],
                 [3.14,3.24,3.34]]
                ])

        test_id = np.array([
            [[3],[1],[4],[2],[5]],
            [[3],[4],[1],[2],[5]],
            [[3],[1],[2],[4],[5]],
            [[5],[1],[2],[4],[3]]
            ])

        test_im = np.array([
            [[0,0,0],
             [1,0,0],
             [0,1,0],
             [0,0,1],
             [1,1,1]],
            [[0,0,0],
             [-1,-1,-1],
             [-13,1,1],
             [2,3,4],
             [1,1,0]],
            [[-1,1,-1],
             [0,0,0],
             [2,3,2],
             [1,-1,1],
             [0,-10,0]],
            [[-1,1,-1],
             [7,34,0],
             [2,3,-2],
             [1,-1,1],
             [0,-10,0]]
            ])

        test_box = np.array(
            [10.0,10.0,10.0]
            )

        pos_ds[:,:,:] = test_pos
        id_ds[:,:,:] = test_id
        im_ds[:,:,:] = test_im
        box_ds[:] = test_box

        h5_fh.close()

        self.h5_fh = h5py.File("test.h5", 'r')

    def test_folded(self):
        self.assertTrue(np.allclose(h5md_pos(self.h5_fh,
                                         np.arange(0,4),
                                         folded=True),
                        reference_data.pos_folded))

    def test_unfolded(self):
        self.assertTrue(np.allclose(h5md_pos(self.h5_fh,
                                         np.arange(0,4),
                                         folded=False),
                        reference_data.pos_unfolded))

    @classmethod
    def tearDownClass(self):
        os.remove("test.h5")


if __name__ == "__main__": 
    suite = unittest.TestLoader().loadTestsFromTestCase(H5md_Pos)
    unittest.TextTestRunner(verbosity=2).run(suite)
