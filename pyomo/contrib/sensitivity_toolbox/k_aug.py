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

# ______________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and
# Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
# rights in this software.
# This software is distributed under the 3-clause BSD License
# ______________________________________________________________________________
import os
from pyomo.environ import SolverFactory
from pyomo.common.tempfiles import TempfileManager


debug_dir = "kaug_debug"
gjh_dir = "GJH"
# These are files we would like to save from a call to k_aug
# or dot_sens. Other files generated will still be deleted,
# but not saved on the K_augInterface object.
known_files = [
    "dsdp_in_.in",
    "conorder.txt",
    "timings_k_aug_dsdp.txt",
    "dot_out.out",
    "delta_p.out",
    "timings_dot_driver_dsdp.txt",
    os.path.join(debug_dir, "kkt.in"),
    os.path.join(gjh_dir, "gradient_f_print.txt"),
    os.path.join(gjh_dir, "A_print.txt"),
]


class InTempDir(object):
    def __init__(self, suffix=None, prefix=None, dir=None):
        self._suffix = suffix
        self._prefix = prefix
        self._dir = dir

    def __enter__(self):
        self._cwd = os.getcwd()
        # Add a new context
        TempfileManager.push()
        # Create a new tempdir in this context
        self._tempdir = TempfileManager.create_tempdir(
            suffix=self._suffix, prefix=self._prefix, dir=self._dir
        )
        os.chdir(self._tempdir)

    def __exit__(self, ex_type, ex_val, ex_bt):
        os.chdir(self._cwd)
        TempfileManager.pop()


class K_augInterface(object):
    """
    k_aug and dot_sens store information in the user's filesystem,
    some of which is mandatory for subsequent calls.
    This class ensures that calls to these executables happen in
    temporary directories. The resulting files are immediately read
    and cached as attributes of this object, and the temporary
    directories deleted. If we have cached files that can be used
    by a subsequent call to k_aug or dot_sens, we write them just
    before calling the executable, and they are deleted along with
    the temporary directory.

    NOTE: only covers dsdp_mode for now.
    """

    def __init__(self, k_aug=None, dot_sens=None):
        # The user may want to use their own k_aug/dot_sens solver
        # objects, i.e. with custom options or executable.
        if k_aug is None:
            k_aug = SolverFactory("k_aug")

            # TODO: Remove this if/when we support RH mode.
            k_aug.options["dsdp_mode"] = ""
        if dot_sens is None:
            dot_sens = SolverFactory("dot_sens")

            # TODO: Remove this if/when we support RH mode.
            dot_sens.options["dsdp_mode"] = ""

        if k_aug.available():
            self._k_aug = k_aug
        else:
            raise RuntimeError("k_aug is not available.")
        if dot_sens.available():
            self._dot_sens = dot_sens
        else:
            raise RuntimeError("dot_sens is not available")

        self.data = {fname: None for fname in known_files}

    def k_aug(self, model, **kwargs):
        with InTempDir():
            # Assume that k_aug doesn't need any files as inputs
            # (except the nl file, which is handled by solve).

            # Call k_aug
            results = self._k_aug.solve(model, **kwargs)

            # Read any files we expect as output
            for fname in known_files:
                if os.path.exists(fname):
                    with open(fname, "r") as fp:
                        self.data[fname] = fp.read()

        return results

    def dot_sens(self, model, **kwargs):
        with InTempDir():
            # Write cached files, some of which dot_sens may use as input
            for fname, contents in self.data.items():
                if contents is not None:
                    with open(fname, "w") as fp:
                        fp.write(contents)

            # Call dot_sens
            results = self._dot_sens.solve(model, **kwargs)

            # Read expected files, some of which may have been created
            # or overwritten by dot_sens.
            for fname in known_files:
                if os.path.exists(fname):
                    with open(fname, "r") as fp:
                        self.data[fname] = fp.read()

        return results

    def set_k_aug_options(self, **options):
        for key, val in options.items():
            self._k_aug.options[key] = val

    def set_dot_sens_options(self, **options):
        for key, val in options.items():
            self._dot_sens.options[key] = val
