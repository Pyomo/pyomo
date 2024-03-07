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

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np
from math import pi, sqrt

fig = plt.figure()
ax = fig.gca(projection='3d')
R_ = np.arange(0.25, 10, 0.25)
H_ = np.arange(0.25, 10, 0.25)
R, H = np.meshgrid(R_, H_)
Z = 2 * pi * R * (R + H)
surf = ax.plot_surface(
    R, H, Z, rstride=1, cstride=1, cmap=cm.hot, linewidth=0, antialiased=False
)
ax.set_xlabel("r")
ax.set_ylabel("h")
ax.set_zlim(0, 1200)

ax.zaxis.set_major_locator(LinearLocator(10))
# ax.zaxis.set_major_formatter(FormatStrFormatter(' %.02f'))

# fig.colorbar(surf, shrink=0.5, aspect=5)

H_ = 355 / (pi * R_ * R_)
valid = np.where(H_ < 10.1)
Z_ = R_ + H_
Z_ = 2 * pi * R_ * Z_
ax.plot(R_[valid], H_[valid], Z_[valid], label='parametric curve')

plt.show()
