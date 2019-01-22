"""
Visualize the trajectory from example01.py. Assuming it was run with these
options:

    -ts_save_trajectory=1
    -ts_trajectory_type=visualization

That should put results from each time step in the directory Visualization-data

To run this you need $PETSC_DIR/lib/petsc/bin/ in PYTHONPATH
> export PYTHONPATH=$PYTHONPATH:$PETSC_DIR/lib/petsc/bin/
"""


from __future__ import division  # No integer division
from __future__ import print_function  # Python 3 style print

try:
    import PetscBinaryIOTrajectory as pbt
except:
    pbt = None
    print("Could not find PetscBinaryIOTrajectory.py.\n"
          "Make sure the PETSc python files are in your PYTHONPATH\n"
          "Try > export PYTHONPATH=$PYTHONPATH:$PETSC_DIR/lib/petsc/bin/")

if __name__ == '__main__':
    if pbt is None:
        exit()
    (t,v,names) = pbt.ReadTrajectory("Visualization-data")
    names = [
        "y1", "y2", "y3", "y4", "y5", "y6",
        "r1", "r2", "r3", "r4", "r5",
        "Fin"]
    pbt.PlotTrajectories(t,v,names,["y1", "y3", "y6"])
    pbt.PlotTrajectories(t,v,names,["y5"])
    pbt.PlotTrajectories(t,v,names,["y2"])
    pbt.PlotTrajectories(t,v,names,["y4"])
