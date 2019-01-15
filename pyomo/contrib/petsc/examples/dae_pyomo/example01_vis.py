"""
Visualize the trajectory from example01.py. Assuming it was run with these
options:

    -ts_save_trajectory=1
    -ts_trajectory_type=visualization

That should put results from each time step in the directory Visualization-data

To run this you need $PETSC_DIR/lib/petsc/bin/ in PYTHONPATH
"""

#commented out because you need some files out of PETSc to run this.
"""
from __future__ import division  # No integer division
from __future__ import print_function  # Python 3 style print
import PetscBinaryIOTrajectory as pbt

if __name__ == '__main__':
    (t,v,names) = pbt.ReadTrajectory("Visualization-data")
    names = [
        "y1", "y2", "y3", "y4", "y5", "y6",
        "r1", "r2", "r3", "r4", "r5",
        "Fin"]
    pbt.PlotTrajectories(t,v,names,["y1", "y2", "y3", "y4", "y5", "y6"])
"""
