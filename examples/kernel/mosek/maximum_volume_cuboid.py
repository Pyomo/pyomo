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

from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
import itertools
import numpy as np

import pyomo.kernel as pmo

# Vertices of a regular icosahedron with edge length 2
f = (1 + np.sqrt(5)) / 2
icosahedron = np.array(
    [
        [0, 1, f],
        [0, -1, f],
        [0, 1, -f],
        [0, -1, -f],
        [1, f, 0],
        [1, -f, 0],
        [-1, f, 0],
        [-1, -f, 0],
        [f, 0, 1],
        [-f, 0, 1],
        [f, 0, -1],
        [-f, 0, -1],
    ]
)
print(f"Volume of the icosahedron = {2.18169699*8}")


def convex_hull_constraint(model, p_v, c_v, v_index):
    A = np.vstack(
        (
            np.eye(len(model.p)),  # p-variable coefficients
            np.diag(c_v),  # x-variable coefficients
            p_v,
        )
    )  # u-variable coefficients
    A = np.transpose(A)
    # Sum(u_i) = 1
    row = [0] * len(list(model.p) + list(model.x)) + [1] * len(model.u[v_index])
    A = np.vstack([A, row])
    # x
    var_vector = list(model.p) + list(model.x) + list(model.u[v_index])
    # b
    b = np.array([0] * A.shape[0])
    b[-1] = 1

    # Matrix constraint ( Ax = b )
    return pmo.matrix_constraint(A, rhs=b, x=var_vector)


def pyomo_maxVolCuboid(vertices):
    m, n = len(vertices), len(vertices[0])

    model = pmo.block()
    model.cuboid_vertices = list(itertools.product([0, 1], repeat=n))

    # Variables
    model.x = pmo.variable_list(pmo.variable(lb=0.0) for i in range(n))
    model.p = pmo.variable_list(pmo.variable() for i in range(n))
    model.t = pmo.variable()

    model.u = pmo.variable_list(
        pmo.variable_list(pmo.variable(lb=0.0) for j in range(m)) for i in range(2**n)
    )

    # Maximize: (volume_of_cuboid)**1/n
    model.cuboid_volume = pmo.objective(model.t, sense=-1)
    # Cone: Geometric-mean conic constraint
    model.geo_cone = pmo.conic.primal_geomean(r=model.x, x=model.t)

    # K : Convex hull formed by the vertices of the polyhedron
    model.conv_hull = pmo.constraint_list()
    for i in range(2**n):
        model.conv_hull.append(
            convex_hull_constraint(model, vertices, model.cuboid_vertices[i], i)
        )

    opt = pmo.SolverFactory("mosek")
    result = opt.solve(model, tee=True)

    _x = np.array([x.value for x in model.x])
    _p = np.array([p.value for p in model.p])
    cuboid_vertices = np.array([_p + e * _x for e in model.cuboid_vertices])
    return cuboid_vertices


cuboid = pyomo_maxVolCuboid(icosahedron)


# Make an interactive 3-D plot


def inscribed_cuboid_plot(icosahedron, cuboid):
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection="3d")

    ico_hull = ConvexHull(icosahedron)
    for s in ico_hull.simplices:
        tri = Poly3DCollection([icosahedron[s]])
        tri.set_edgecolor('black')
        tri.set_alpha(0.3)
        tri.set_facecolor('red')
        ax.add_collection3d(tri)
        ax.scatter(
            icosahedron[:, 0], icosahedron[:, 1], icosahedron[:, 2], color='darkred'
        )

    cub_hull = ConvexHull(cuboid)
    for s in cub_hull.simplices:
        tri = Poly3DCollection([cuboid[s]])
        # tri.set_edgecolor('black')
        tri.set_alpha(0.8)
        tri.set_facecolor('blue')
        ax.add_collection3d(tri)

    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_zlim(-2, 2)
    plt.show()


inscribed_cuboid_plot(icosahedron, cuboid)
