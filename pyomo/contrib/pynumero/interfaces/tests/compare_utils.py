#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

def check_vectors_specific_order(tst, v1, v1order, v2, v2order):
    tst.assertEqual(len(v1), len(v1order))
    tst.assertEqual(len(v2), len(v2order))
    tst.assertEqual(len(v1), len(v2))
    v2map = {s:i for i,s in enumerate(v2order)}
    for i,s in enumerate(v1order):
        tst.assertEqual(v1[i], v2[v2map[s]])

def check_sparse_matrix_specific_order(tst, m1, m1rows, m1cols, m2, m2rows, m2cols):
    tst.assertEqual(m1.shape[0], len(m1rows))
    tst.assertEqual(m1.shape[1], len(m1cols))
    tst.assertEqual(m2.shape[0], len(m2rows))
    tst.assertEqual(m2.shape[1], len(m2cols))
    tst.assertEqual(len(m1rows), len(m2rows))
    tst.assertEqual(len(m1cols), len(m2cols))

    m1c = m1.todense()
    m2c = np.zeros((len(m2rows), len(m2cols)))
    rowmap = [m2rows.index(x) for x in m1rows]
    colmap = [m2cols.index(x) for x in m1cols]
    for i in range(len(m1rows)):
        for j in range(len(m1cols)):
            m2c[i,j] = m2[rowmap[i], colmap[j]]

    #print(m1c)
    #print(m2c)
    tst.assertTrue(np.array_equal(m1c, m2c))
