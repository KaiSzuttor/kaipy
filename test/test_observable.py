#!/usr/bin/env python


import unittest
import numpy as np
from kaipy.observable import second_legendre, rg2

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

        
class Rg2_test(unittest.TestCase):
    def setUp(self):
        self.number_of_coordinates = 10000
        self.coordinates = np.zeros((self.number_of_coordinates,3))
        self.coordinates[:,0] = np.linspace(0,12,self.number_of_coordinates)

    def test_function(self):
        self.assertAlmostEqual(rg2(self.coordinates),12,delta=0.01)

if __name__ == "__main__": 
    unittest.main()
