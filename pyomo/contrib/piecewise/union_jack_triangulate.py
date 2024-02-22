#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2024
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import itertools
from math import factorial
import time

# This implements the J1 "Union Jack" triangulation (Todd 77) as explained by
# Vielma 2010.

# Triangulate {0, ..., K}^n for even K using the J1 triangulation.
def triangulate(K, n):
    if K % 2 != 0:
        raise ValueError("K must be even")
    # 1, 3, ..., K - 1
    axis_odds = range(1, K, 2)
    V_0 = itertools.product(axis_odds, repeat=n)
    big_iterator = itertools.product(V_0, 
                                     itertools.permutations(range(0, n), n), 
                                     itertools.product((-1, 1), repeat=n))
    J1 = []
    for v_0, pi, s in big_iterator:
        simplex = []
        current = list(v_0)
        simplex.append(current)
        for i in range(0, n):
            current = current.copy()
            current[pi[i]] += s[pi[i]]
            simplex.append(current)
        J1.append(simplex) 
    return J1

if __name__ == '__main__':
    # do some tests. TODO move to real test file
    start0 = time.time()
    small_2d = triangulate(2, 2)
    elapsed0 = time.time() - start0
    print(f"triangulated small_2d in {elapsed0} sec.")
    assert len(small_2d) == 8
    assert small_2d == [[[1, 1], [0, 1], [0, 0]], 
                        [[1, 1], [0, 1], [0, 2]], 
                        [[1, 1], [2, 1], [2, 0]], 
                        [[1, 1], [2, 1], [2, 2]], 
                        [[1, 1], [1, 0], [0, 0]], 
                        [[1, 1], [1, 2], [0, 2]], 
                        [[1, 1], [1, 0], [2, 0]], 
                        [[1, 1], [1, 2], [2, 2]]]
    start1 = time.time()
    bigger_2d = triangulate(4, 2)
    elapsed1 = time.time() - start1
    print(f"triangulated bigger_2d in {elapsed1} sec.")
    assert len(bigger_2d) == 32

    start2 = time.time()
    medium_3d = triangulate(12, 3)
    elapsed2 = time.time() - start2
    print(f"triangulated medium_3d in {elapsed2} sec.")
    # A J1 triangulation of {0, ..., K}^n has K^n * n! simplices
    assert len(medium_3d) == 12**3 * factorial(3)

    start3 = time.time()
    big_4d = triangulate(20, 4)
    elapsed3 = time.time() - start3
    print(f"triangulated big_4d in {elapsed3} sec.")
    assert len(big_4d) == 20**4 * factorial(4)

    print("starting huge_5d")
    start4 = time.time()
    huge_5d = triangulate(10, 5)
    elapsed4 = time.time() - start4
    print(f"triangulated huge_5d in {elapsed4} sec.")

    print("Success")
