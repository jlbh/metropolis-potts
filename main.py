#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 01:15:37 2023

@author: johannes
"""

import numpy as np
from rich import print

class Potts:
    def __init__(self, q, length, temp, mode):
        self.q = q
        self.length = length
        self.temp = temp
        
        if mode == 0: self.lattice = np.zeros((length, length), dtype=int)
        if mode == 1: self.lattice = np.random.randint(0, q, (length, length))

        x_pairs = self.lattice - np.roll(self.lattice, 1, axis=0)
        y_pairs = self.lattice - np.roll(self.lattice, 1, axis=1)
        self.energy =  np.count_nonzero(x_pairs) + np.count_nonzero(y_pairs) - 2*length**2

    def metropolis(self):
        i, j = np.random.randint(0, self.length, 2)
        old_spin = self.lattice[i, j]
        new_spin = np.random.choice(np.delete(np.arange(self.q), old_spin))

        neighbours = np.array([self.lattice[(i+1) % self.length, j], self.lattice[i-1, j],
                               self.lattice[i, (j+1) % self.length], self.lattice[i, j-1]])

        delta_e = np.count_nonzero(neighbours - new_spin) - np.count_nonzero(neighbours - old_spin)

        prob = min(1, np.exp(-delta_e / self.temp))

        if np.random.uniform() < prob:
            self.lattice[i, j] = new_spin
            self.energy += delta_e

def main():
    m = Potts(8, 20, .1, 1)
    
    for t in range(10_000): 
        print(str(m.lattice).replace(' [', '[[').replace(' ', ')]██[color(')
                            .replace('[[', '[color(').replace(']]', ')]'))
        for tt in range(100): 
            m.metropolis()

if __name__ == '__main__': main()
