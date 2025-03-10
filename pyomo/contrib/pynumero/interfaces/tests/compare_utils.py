#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________
from pyomo.contrib.pynumero.dependencies import numpy as np, scipy


def check_vectors_specific_order(tst, v1, v1order, v2, v2order, v1_v2_map=None):
    tst.assertEqual(len(v1), len(v1order))
    tst.assertEqual(len(v2), len(v2order))
    tst.assertEqual(len(v1), len(v2))
    if v1_v2_map is None:
        v2map = {s: v2order.index(s) for s in v1order}
    else:
        v2map = {s: v2order.index(v1_v2_map[s]) for s in v1order}
    for i, s in enumerate(v1order):
        tst.assertAlmostEqual(v1[i], v2[v2map[s]], places=7)


def check_sparse_matrix_specific_order(
    tst,
    m1,
    m1rows,
    m1cols,
    m2,
    m2rows,
    m2cols,
    m1_m2_rows_map=None,
    m1_m2_cols_map=None,
):
    tst.assertEqual(m1.shape[0], len(m1rows))
    tst.assertEqual(m1.shape[1], len(m1cols))
    tst.assertEqual(m2.shape[0], len(m2rows))
    tst.assertEqual(m2.shape[1], len(m2cols))
    tst.assertEqual(len(m1rows), len(m2rows))
    tst.assertEqual(len(m1cols), len(m2cols))

    m1c = m1
    if scipy.sparse.issparse(m1c):
        m1c = m1c.todense()
    m2d = m2
    if scipy.sparse.issparse(m2d):
        m2d = m2d.todense()

    m2c = np.zeros((len(m2rows), len(m2cols)))
    if m1_m2_rows_map is None:
        rowmap = [m2rows.index(x) for x in m1rows]
    else:
        rowmap = [m2rows.index(m1_m2_rows_map[x]) for x in m1rows]
    if m1_m2_cols_map is None:
        colmap = [m2cols.index(x) for x in m1cols]
    else:
        colmap = [m2cols.index(m1_m2_cols_map[x]) for x in m1cols]

    for i in range(len(m1rows)):
        for j in range(len(m1cols)):
            m2c[i, j] = m2d[rowmap[i], colmap[j]]

    for i in range(len(m1rows)):
        for j in range(len(m1cols)):
            tst.assertAlmostEqual(m1c[i, j], m2c[i, j], places=7)
