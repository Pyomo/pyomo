#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2022
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import logging
import glob

from pyomo.common.tempfiles import TempfileManager
from pyomo.solvers.plugins.solvers.ASL import ASL

logger = logging.getLogger('pyomo.contrib.gjh')


def readgjh(fname=None):
    """
    Build objective gradient and constraint Jacobian
    from gjh file written by the ASL gjh 'solver'.

    gjh solver may be called using pyomo and 'keepfiles=True'.
    Enable 'symbolic_solver_labels' as well to write
    .row and col file to get variable mappings.

    Parameters
    ----------
    fname : string, optional
        gjh file name. The default is None.

    Returns
    -------
    g : list
        Current objective gradient.
    J : list
        Current objective Jacobian.
    H : list
        Current objective Hessian.
    variableList : list
        Variables as defined by *.col file.
    constraintList : list
        Constraints as defined by *.row file.

    """
    if fname is None:
        files = list(glob.glob("*.gjh"))
        fname = files.pop(0)
        if len(files) > 1:
            print("WARNING: More than one gjh file in current directory")
            print("  Processing: %s\nIgnoring: %s" % (fname, '\n\t\t'.join(files)))

    with open(fname, "r") as f:
        line = "dummy_str"
        # Skip top lines until g value is reached
        while line != "param g :=\n":
            line = f.readline()

        line = f.readline()
        g = []
        # ; is the escape character
        while line[0] != ';':
            """
            When printed via ampl interface:
            ampl: display g;
            g [*] :=
                         1   0.204082
                         2   0.367347
                         3   0.44898
                         4   0.44898
                         5   0.244898
                         6  -0.173133
                         7  -0.173133
                         8  -0.0692532
                         9   0.0692532
                        10   0.346266
                        ;
            """
            index = int(line.split()[0]) - 1
            value = float(line.split()[1])
            g.append([index, value])
            line = f.readline()

        # Skip lines until J value is reached
        while line != "param J :=\n":
            line = f.readline()

        line = f.readline()
        J = []
        while line[0] != ';':
            """
            When printed via ampl interface:
            ampl: display J;
            J [*,*]
                :         1             2           3          4           5            6
                 :=
                1    -0.434327       0.784302     .          .           .          -0.399833
                2     2.22045e-16     .          1.46939     .           .          -0.831038
                3     0.979592        .           .         1.95918      .          -0.9596
                4     1.79592         .           .          .          2.12245     -0.692532
                5     0.979592        .           .          .           .           0
                6      .            -0.0640498   0.545265    .           .            .
                7      .             0.653061     .         1.14286      .            .
                8      .             1.63265      .          .          1.63265       .
                9      .             1.63265      .          .           .            .
                10     .              .          0.262481   0.262481     .            .
                11     .              .          1.14286     .          0.653061      .
                12     .              .          1.95918     .           .            .
                13     .              .           .         0.545265   -0.0640498     .
                14     .              .           .         1.95918      .            .
                15     .              .           .          .          1.63265       .
                16     .              .           .          .           .          -1

                :        7           8           9          10       :=
                1     0.399833     .           .          .
                2      .          0.831038     .          .
                3      .           .          0.9596      .
                4      .           .           .         0.692532
                6    -0.799667    0.799667     .          .
                7    -1.38506      .          1.38506     .
                8    -1.33278      .           .         1.33278
                9     0            .           .          .
                10     .         -0.9596      0.9596      .
                11     .         -1.38506      .         1.38506
                12     .          0            .          .
                13     .           .         -0.799667   0.799667
                14     .           .          0           .
                15     .           .           .         0
                16    1            .           .          .
                17   -1           1            .          .
                18     .         -1           1           .
                19     .           .         -1          1
            ;
            """
            if line[0] == '[':
                # Jacobian row index
                row = int(''.join(filter(str.isdigit, line))) - 1
                line = f.readline()

            column = int(line.split()[0]) - 1
            value = float(line.split()[1])
            J.append([row, column, value])
            line = f.readline()

        while line != "param H :=\n":
            line = f.readline()

        line = f.readline()
        H = []
        while line[0] != ';':
            """
            When printed via ampl interface:
            ampl: display H;
                H [*,*]
                :       1           2           3           4           5           6         :=
                1      .         0.25         .           .           .         -0.35348
                2     0.25        .          0.25         .           .         -0.212088
                3      .         0.25         .          0.25         .           .
                4      .          .          0.25         .          0.25         .
                5      .          .           .          0.25         .           .
                6    -0.35348   -0.212088     .           .           .         -0.0999584
                7     0.35348   -0.212088   -0.35348      .           .          0.0999584
                8      .         0.424176   -0.070696   -0.424176     .           .
                9      .          .          0.424176    0.070696   -0.424176     .
                10     .          .           .          0.35348     0.424176     .

                :        7            8           9          10        :=
                1     0.35348       .           .           .
                2    -0.212088     0.424176     .           .
                3    -0.35348     -0.070696    0.424176     .
                4      .          -0.424176    0.070696    0.35348
                5      .            .         -0.424176    0.424176
                6     0.0999584     .           .           .
                7    -0.299875     0.199917     .           .
                8     0.199917    -0.439817    0.2399       .
                9      .           0.2399     -0.439817    0.199917
                10     .            .          0.199917   -0.199917
                ;
            """
            if line[0] == '[':
                # Hessian row index
                row = int(''.join(filter(str.isdigit, line))) - 1
                line = f.readline()

            column = int(line.split()[0]) - 1
            value = float(line.split()[1])
            H.append([row, column, value])
            line = f.readline()

    with open(fname[:-3] + 'col', 'r') as f:
        data = f.read()
        variableList = data.split()

    with open(fname[:-3] + 'row', 'r') as f:
        data = f.read()
        constraintList = data.split()

    return g, J, H, variableList, constraintList


class GJHSolver(ASL):
    """
    An interface to the AMPL GJH "solver" for evaluating a model at a
    point.
    """

    def __init__(self, **kwds):
        kwds['type'] = 'gjh'
        kwds['symbolic_solver_labels'] = True
        super().__init__(**kwds)
        self.options.solver = 'gjh'
        self._metasolver = False

    # A hackish way to hold on to the model so that we can parse the
    # results.
    def _initialize_callbacks(self, model):
        self._model = model
        self._model._gjh_info = None
        super()._initialize_callbacks(model)

    def _presolve(self, *args, **kwds):
        super()._presolve(*args, **kwds)
        self._gjh_file = self._soln_file[:-3] + 'gjh'
        TempfileManager.add_tempfile(self._gjh_file, exists=False)

    def _postsolve(self):
        #
        # TODO: We should return the information using a better data
        # structure (ComponentMap?) so that the GJH solver does not need
        # to be called with symbolic_solver_labels=True
        #
        self._model._gjh_info = readgjh(self._gjh_file)
        self._model = None
        return super()._postsolve()
