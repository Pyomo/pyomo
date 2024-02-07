"""
Test objects for construction of PyROS ConfigDict.
"""


import unittest

from pyomo.core.base import (
    ConcreteModel,
    Var,
    _VarData,
)
from pyomo.core.base.param import Param, _ParamData
from pyomo.contrib.pyros.config import (
    InputDataStandardizer,
    mutable_param_validator,
)


class testInputDataStandardizer(unittest.TestCase):
    """
    Test standardizer method for Pyomo component-type inputs.
    """

    def test_single_component_data(self):
        """
        Test standardizer works for single component
        data-type entry.
        """
        mdl = ConcreteModel()
        mdl.v = Var([0, 1])

        standardizer_func = InputDataStandardizer(Var, _VarData)

        standardizer_input = mdl.v[0]
        standardizer_output = standardizer_func(standardizer_input)

        self.assertIsInstance(
            standardizer_output,
            list,
            msg=(
                "Standardized output should be of type list, "
                f"but is of type {standardizer_output.__class__.__name__}."
            ),
        )
        self.assertEqual(
            len(standardizer_output),
            1,
            msg="Length of standardizer output is not as expected.",
        )
        self.assertIs(
            standardizer_output[0],
            mdl.v[0],
            msg=(
                f"Entry {standardizer_output[0]} (id {id(standardizer_output[0])}) "
                "is not identical to "
                f"input component data object {mdl.v[0]} "
                f"(id {id(mdl.v[0])})"
            ),
        )

    def test_standardizer_indexed_component(self):
        """
        Test component standardizer works on indexed component.
        """
        mdl = ConcreteModel()
        mdl.v = Var([0, 1])

        standardizer_func = InputDataStandardizer(Var, _VarData)

        standardizer_input = mdl.v
        standardizer_output = standardizer_func(standardizer_input)

        self.assertIsInstance(
            standardizer_output,
            list,
            msg=(
                "Standardized output should be of type list, "
                f"but is of type {standardizer_output.__class__.__name__}."
            ),
        )
        self.assertEqual(
            len(standardizer_output),
            2,
            msg="Length of standardizer output is not as expected.",
        )
        enum_zip = enumerate(zip(standardizer_input.values(), standardizer_output))
        for idx, (input, output) in enum_zip:
            self.assertIs(
                input,
                output,
                msg=(
                    f"Entry {input} (id {id(input)}) "
                    "is not identical to "
                    f"input component data object {output} "
                    f"(id {id(output)})"
                ),
            )

    def test_standardizer_multiple_components(self):
        """
        Test standardizer works on sequence of components.
        """
        mdl = ConcreteModel()
        mdl.v = Var([0, 1])
        mdl.x = Var(["a", "b"])

        standardizer_func = InputDataStandardizer(Var, _VarData)

        standardizer_input = [mdl.v[0], mdl.x]
        standardizer_output = standardizer_func(standardizer_input)
        expected_standardizer_output = [mdl.v[0], mdl.x["a"], mdl.x["b"]]

        self.assertIsInstance(
            standardizer_output,
            list,
            msg=(
                "Standardized output should be of type list, "
                f"but is of type {standardizer_output.__class__.__name__}."
            ),
        )
        self.assertEqual(
            len(standardizer_output),
            len(expected_standardizer_output),
            msg="Length of standardizer output is not as expected.",
        )
        enum_zip = enumerate(zip(expected_standardizer_output, standardizer_output))
        for idx, (input, output) in enum_zip:
            self.assertIs(
                input,
                output,
                msg=(
                    f"Entry {input} (id {id(input)}) "
                    "is not identical to "
                    f"input component data object {output} "
                    f"(id {id(output)})"
                ),
            )

    def test_standardizer_invalid_duplicates(self):
        """
        Test standardizer raises exception if input contains duplicates
        and duplicates are not allowed.
        """
        mdl = ConcreteModel()
        mdl.v = Var([0, 1])
        mdl.x = Var(["a", "b"])

        standardizer_func = InputDataStandardizer(Var, _VarData, allow_repeats=False)

        exc_str = r"Standardized.*list.*contains duplicate entries\."
        with self.assertRaisesRegex(ValueError, exc_str):
            standardizer_func([mdl.x, mdl.v, mdl.x])

    def test_standardizer_invalid_type(self):
        """
        Test standardizer raises exception as expected
        when input is of invalid type.
        """
        standardizer_func = InputDataStandardizer(Var, _VarData)

        exc_str = r"Input object .*is not of valid component type.*"
        with self.assertRaisesRegex(TypeError, exc_str):
            standardizer_func(2)

    def test_standardizer_iterable_with_invalid_type(self):
        """
        Test standardizer raises exception as expected
        when input is an iterable with entries of invalid type.
        """
        mdl = ConcreteModel()
        mdl.v = Var([0, 1])
        standardizer_func = InputDataStandardizer(Var, _VarData)

        exc_str = r"Input object .*entry of iterable.*is not of valid component type.*"
        with self.assertRaisesRegex(TypeError, exc_str):
            standardizer_func([mdl.v, 2])

    def test_standardizer_invalid_str_passed(self):
        """
        Test standardizer raises exception as expected
        when input is of invalid type str.
        """
        standardizer_func = InputDataStandardizer(Var, _VarData)

        exc_str = r"Input object .*is not of valid component type.*"
        with self.assertRaisesRegex(TypeError, exc_str):
            standardizer_func("abcd")

    def test_standardizer_invalid_unintialized_params(self):
        """
        Test standardizer raises exception when Param with
        uninitialized entries passed.
        """
        standardizer_func = InputDataStandardizer(
            ctype=Param, cdatatype=_ParamData, ctype_validator=mutable_param_validator
        )

        mdl = ConcreteModel()
        mdl.p = Param([0, 1])

        exc_str = r"Length of .*does not match that of.*index set"
        with self.assertRaisesRegex(ValueError, exc_str):
            standardizer_func(mdl.p)

    def test_standardizer_invalid_immutable_params(self):
        """
        Test standardizer raises exception when immutable
        Param object(s) passed.
        """
        standardizer_func = InputDataStandardizer(
            ctype=Param, cdatatype=_ParamData, ctype_validator=mutable_param_validator
        )

        mdl = ConcreteModel()
        mdl.p = Param([0, 1], initialize=1)

        exc_str = r"Param object with name .*immutable"
        with self.assertRaisesRegex(ValueError, exc_str):
            standardizer_func(mdl.p)

    def test_standardizer_valid_mutable_params(self):
        """
        Test Param-like standardizer works as expected for sequence
        of valid mutable Param objects.
        """
        mdl = ConcreteModel()
        mdl.p1 = Param([0, 1], initialize=0, mutable=True)
        mdl.p2 = Param(["a", "b"], initialize=1, mutable=True)

        standardizer_func = InputDataStandardizer(
            ctype=Param, cdatatype=_ParamData, ctype_validator=mutable_param_validator
        )

        standardizer_input = [mdl.p1[0], mdl.p2]
        standardizer_output = standardizer_func(standardizer_input)
        expected_standardizer_output = [mdl.p1[0], mdl.p2["a"], mdl.p2["b"]]

        self.assertIsInstance(
            standardizer_output,
            list,
            msg=(
                "Standardized output should be of type list, "
                f"but is of type {standardizer_output.__class__.__name__}."
            ),
        )
        self.assertEqual(
            len(standardizer_output),
            len(expected_standardizer_output),
            msg="Length of standardizer output is not as expected.",
        )
        enum_zip = enumerate(zip(expected_standardizer_output, standardizer_output))
        for idx, (input, output) in enum_zip:
            self.assertIs(
                input,
                output,
                msg=(
                    f"Entry {input} (id {id(input)}) "
                    "is not identical to "
                    f"input component data object {output} "
                    f"(id {id(output)})"
                ),
            )


if __name__ == "__main__":
    unittest.main()
