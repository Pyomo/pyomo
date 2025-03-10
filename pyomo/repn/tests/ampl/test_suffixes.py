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
#
# Test that EXPORT type Suffix components work with the NL writer
#

import os

import pyomo.common.unittest as unittest

from pyomo.common.fileutils import this_file_dir
from pyomo.common.tempfiles import TempfileManager

from pyomo.opt import ProblemFormat
from pyomo.environ import (
    ConcreteModel,
    Suffix,
    Var,
    Objective,
    Constraint,
    SOSConstraint,
    sum_product,
)
from ..nl_diff import load_and_compare_nl_baseline

currdir = this_file_dir()


class SuffixTester(object):
    @classmethod
    def setUpClass(cls):
        cls.context = TempfileManager.new_context()
        cls.tempdir = cls.context.create_tempdir()

    @classmethod
    def tearDownClass(cls):
        cls.context.release()

    # test that EXPORT suffixes on variables,
    # constraints, objectives, and models
    # will end up in the NL file with integer tags
    def test_EXPORT_suffixes_int(self):
        model = ConcreteModel()
        model.junk = Suffix(direction=Suffix.EXPORT, datatype=Suffix.INT)
        model.junk_inactive = Suffix(direction=Suffix.EXPORT, datatype=Suffix.INT)

        model.x = Var()
        model.junk.set_value(model.x, 1)
        model.junk_inactive.set_value(model.x, 1)

        model.y = Var([1, 2], dense=True)
        model.junk.set_value(model.y, 2)
        model.junk_inactive.set_value(model.y, 2)

        model.obj = Objective(expr=model.x + sum_product(model.y))
        model.junk.set_value(model.obj, 3)
        model.junk_inactive.set_value(model.obj, 3)

        model.conx = Constraint(expr=model.x >= 1)
        model.junk.set_value(model.conx, 4)
        model.junk_inactive.set_value(model.conx, 4)

        model.cony = Constraint([1, 2], rule=lambda model, i: model.y[i] >= 1)
        model.junk.set_value(model.cony, 5)
        model.junk_inactive.set_value(model.cony, 5)

        model.junk.set_value(model, 6)
        model.junk_inactive.set_value(model, 6)

        # This one should NOT end up in the NL file
        model.junk_inactive.deactivate()

        _test = os.path.join(self.tempdir, "EXPORT_suffixes.test.nl")
        model.write(
            filename=_test,
            format=self.nl_version,
            io_options={"symbolic_solver_labels": False, "file_determinism": 1},
        )
        _base = os.path.join(currdir, "EXPORT_suffixes_int.baseline.nl")
        self.assertEqual(*load_and_compare_nl_baseline(_base, _test))

    # test that EXPORT suffixes on variables,
    # constraints, objectives, and models
    # will end up in the NL file with floating point tags
    def test_EXPORT_suffixes_float(self):
        model = ConcreteModel()
        model.junk = Suffix(direction=Suffix.EXPORT, datatype=Suffix.FLOAT)
        model.junk_inactive = Suffix(direction=Suffix.EXPORT, datatype=Suffix.FLOAT)

        model.x = Var()
        model.junk.set_value(model.x, 1)
        model.junk_inactive.set_value(model.x, 1)

        model.y = Var([1, 2], dense=True)
        model.junk.set_value(model.y, 2)
        model.junk_inactive.set_value(model.y, 2)

        model.obj = Objective(expr=model.x + sum_product(model.y))
        model.junk.set_value(model.obj, 3)
        model.junk_inactive.set_value(model.obj, 3)

        model.conx = Constraint(expr=model.x >= 1)
        model.junk.set_value(model.conx, 4)
        model.junk_inactive.set_value(model.conx, 4)

        model.cony = Constraint([1, 2], rule=lambda model, i: model.y[i] >= 1)
        model.junk.set_value(model.cony, 5)
        model.junk_inactive.set_value(model.cony, 5)

        model.junk.set_value(model, 6)
        model.junk_inactive.set_value(model, 6)

        # This one should NOT end up in the NL file
        model.junk_inactive.deactivate()

        _test = os.path.join(self.tempdir, "EXPORT_suffixes.test.nl")
        model.write(
            filename=_test,
            format=self.nl_version,
            io_options={"symbolic_solver_labels": False, "file_determinism": 1},
        )
        _base = os.path.join(currdir, "EXPORT_suffixes_float.baseline.nl")
        self.assertEqual(*load_and_compare_nl_baseline(_base, _test))

    # Test that user defined ref suffixes fail to
    # merge with those created from translating the SOSConstraint
    # component when variables get assigned duplicate values for ref
    def test_EXPORT_suffixes_with_SOSConstraint_duplicateref(self):
        model = ConcreteModel()
        model.ref = Suffix(direction=Suffix.EXPORT, datatype=Suffix.INT)
        model.y = Var([1, 2, 3])
        model.obj = Objective(expr=sum_product(model.y))

        # The NL writer will convert this constraint to ref and sosno
        # suffixes on model.y
        model.sos_con = SOSConstraint(var=model.y, index=[1, 2, 3], sos=1)

        for i, val in zip([1, 2, 3], [11, 12, 13]):
            model.ref.set_value(model.y[i], val)

        with self.assertRaisesRegex(
            RuntimeError,
            "NL file writer does not allow both manually "
            "declared 'ref' suffixes as well as SOSConstraint ",
        ):
            model.write(
                filename=os.path.join(self.tempdir, "junk.nl"),
                format=self.nl_version,
                io_options={"symbolic_solver_labels": False},
            )

    # Test that user defined sosno suffixes fail to
    # merge with those created from translating the SOSConstraint
    # component when variables get assigned duplicate values for sosno
    def test_EXPORT_suffixes_with_SOSConstraint_duplicatesosno(self):
        model = ConcreteModel()
        model.sosno = Suffix(direction=Suffix.EXPORT, datatype=Suffix.INT)
        model.y = Var([1, 2, 3])
        model.obj = Objective(expr=sum_product(model.y))

        # The NL writer will convert this constraint to ref and sosno
        # suffixes on model.y
        model.sos_con = SOSConstraint(var=model.y, index=[1, 2, 3], sos=1)

        for i in [1, 2, 3]:
            model.sosno.set_value(model.y[i], -1)

        with self.assertRaisesRegex(
            RuntimeError,
            "NL file writer does not allow both manually "
            "declared 'sosno' suffixes as well as SOSConstraint ",
        ):
            model.write(
                filename=os.path.join(self.tempdir, "junk.nl"),
                format=self.nl_version,
                io_options={"symbolic_solver_labels": False},
            )

    # Test that user defined sosno suffixes fail to
    # merge with those created from translating the SOSConstraint
    # component when variables get assigned duplicate values for sosno
    def test_EXPORT_suffixes_no_datatype(self):
        model = ConcreteModel()
        model.sosno = Suffix(direction=Suffix.EXPORT, datatype=None)
        model.y = Var([1, 2, 3])
        model.obj = Objective(expr=sum_product(model.y))

        # The NL writer will convert this constraint to ref and sosno
        # suffixes on model.y
        model.sos_con = SOSConstraint(var=model.y, index=[1, 2, 3], sos=1)

        for i in [1, 2, 3]:
            model.sosno.set_value(model.y[i], -1)

        with self.assertRaisesRegex(
            RuntimeError,
            "NL file writer does not allow both manually "
            "declared 'sosno' suffixes as well as SOSConstraint ",
        ):
            model.write(
                filename=os.path.join(self.tempdir, "junk.nl"),
                format=self.nl_version,
                io_options={"symbolic_solver_labels": False},
            )


class TestSuffix_nlv1(SuffixTester, unittest.TestCase):
    nl_version = 'nl_v1'


class TestSuffix_nlv2(SuffixTester, unittest.TestCase):
    nl_version = 'nl_v2'


if __name__ == "__main__":
    unittest.main()
