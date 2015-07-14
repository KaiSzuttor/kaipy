#!/usr/bin/env python


import unittest
import numpy as np
from kaipy.observable import second_legendre

class Second_legendre_test(unittest.TestCase):
    def test_x(self):
        self.assertEqual(second_legendre(np.array([0,0,0]),
                                         np.array([1,0,0]),
                                         'x'),
                         1.)
        self.assertEqual(second_legendre(np.array([0,0,0]),
                                         np.array([0,1,0]),
                                         'x'),
                         -.5)
    def test_y(self):
        self.assertEqual(second_legendre(np.array([0,0,0]),
                                         np.array([0,1,0]),
                                         'y'),
                         1.)
        self.assertEqual(second_legendre(np.array([0,0,0]),
                                         np.array([0,0,1]),
                                         'y'),
                         -.5)
    def test_z(self):
        self.assertEqual(second_legendre(np.array([0,0,0]),
                                         np.array([0,0,1]),
                                         'z'),
                         1.)
        self.assertEqual(second_legendre(np.array([0,0,0]),
                                         np.array([1,0,0]),
                                         'z'),
                         -.5)

if __name__ == "__main__": 
    unittest.main()
