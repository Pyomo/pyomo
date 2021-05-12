# ______________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and
# Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
# rights in this software.
# This software is distributed under the 3-clause BSD License
# ______________________________________________________________________________
import os
import enum
from pyomo.environ import (
        SolverFactory,
        )
from pyomo.common.tempfiles import TempfileManager


class FileType(enum.Enum):
    K_AUG_INPUT = 1
    K_AUG_OUTPUT = 2
    DOT_SENS_INPUT = 3
    DOT_SENS_OUTPUT = 4
FT = FileType


def _attribute_from_filename(filename):
    base_name = os.path.basename(filename)
    return base_name.split(".")[0]


debug_dir = "kaug_debug"
# These are files we would like to save from a call to k_aug
# or dot_sens. Other files generated will still be deleted,
# but not saved on the K_augInterface object.
known_files = {
        "dsdp_in_.in": {FT.K_AUG_OUTPUT, FT.DOT_SENS_INPUT},
        "conorder.txt": {FT.K_AUG_OUTPUT},
        "timings_k_aug_dsdp": {FT.K_AUG_OUTPUT},
        "col_row.nl": {FT.K_AUG_OUTPUT},
        "col_row.col": {FT.K_AUG_OUTPUT},
        "col_row.row": {FT.K_AUG_OUTPUT},
        os.path.join(debug_dir, "kkt.in"): {FT.K_AUG_OUTPUT},
        "dot_out.out": {FT.DOT_DRIVER_OUTPUT},
        "delta_p.out": {FT.DOT_DRIVER_OUTPUT},
        "timings_dot_driver_dsdp.txt": {FT.DOT_DIRVER_OUTPUT},
        }

file_attr_map = {name: _attribute_from_filename(name) for name in known_files}

k_aug_input_files = [name for name, types in known_files.items()
        if FT.K_AUG_INPUT in types]
k_aug_output_files = [name for name, types in known_files.items()
        if FT.K_AUG_OUTPUT in types]
dot_sens_input_files = [name for name, types in known_files.items()
        if FT.DOT_SENS_INPUT in types]
dot_sens_output_files = [name for name, types in known_files.items()
        if FT.DOT_SENS_OUTPUT in types]


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

        for fname, attr in file_attr_map.items():
            self.__setattr__(attr, None)

    def k_aug(self, model, **kwargs):
        try:
            # Create a tempdir and descend into it
            cwd = os.getcwd()
            tempdir = TempfileManager.create_tempdir(dir=cwd)
            os.chdir(tempdir)

            # Write any files k_aug may use as input
            for fname in k_aug_input_files:
                attr = file_attr_map[fname]
                contents = self.__getattribute__(attr)
                if contents is not None:
                    with open(fname, "w") as fp:
                        fp.write(contents)

            # Call k_aug
            results = self._k_aug.solve(model, **kwargs)

            # Read any files we expect as output
            for fname in k_aug_output_files:
                attr = file_attr_map[fname]
                if os.path.exists(fname):
                    with open(fname, "r") as fp:
                        self.__setattr__(attr, fp.read())
        finally:
            # Exit tempdir and delete
            os.chdir(cwd)
            TempfileManager.pop()

        return results

    def dot_sens(self, model, **kwargs):
        try:
            # Create a tempdir and descend into it
            cwd = os.getcwd()
            tempdir = TempfileManager.create_tempdir(dir=cwd)
            os.chdir(tempdir)

            # Write any files dot_sens may use as input
            for fname in dot_sens_input_files:
                attr = file_attr_map[fname]
                contents = self.__getattribute__(attr)
                if contents is not None:
                    with open(fname, "w") as fp:
                        fp.write(contents)

            # Call dot_sens
            self._dot_sens.solve(model, **kwargs)

            # Read any files we expect as output
            for fname in dot_sens_output_files:
                attr = file_attr_map[fname]
                if os.path.exists(fname):
                    with open(fname, "r") as fp:
                        self.__setattr__(attr, fp.read())
        finally:
            os.chdir(cwd)
            TempfileManager.pop()

    def set_k_aug_options(self, **options):
        for key, val in options.items():
            self._k_aug.options[key] = val

    def set_dot_sens_options(self, **options):
        for key, val in options.items():
            self._dot_sens.options[key] = val
