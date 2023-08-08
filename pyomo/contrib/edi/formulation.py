#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2023
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#
#  Development of this module was conducted as part of the Institute for
#  the Design of Advanced Energy Systems (IDAES) with support through the
#  Simulation-Based Engineering, Crosscutting Research Program within the
#  U.S. Department of Energy’s Office of Fossil Energy and Carbon Management.
#
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import pyomo
import pyomo.environ as pyo
from pyomo.util.check_units import assert_units_consistent
from pyomo.environ import ConcreteModel
from pyomo.environ import Var, Param, Objective, Constraint, Set
from pyomo.environ import maximize, minimize
from pyomo.environ import units as pyomo_units

egb, egb_available = attempt_import(
    "pyomo.contrib.pynumero.interfaces.external_grey_box"
)
if egb_available:
    from pyomo.contrib.pynumero.interfaces.external_grey_box import (
        ExternalGreyBoxModel,
        ExternalGreyBoxBlock,
    )
else:
    raise ImportError(
        "pyomo.contrib.edi requires pyomo.contrib.pynumero to be installed to enable black box capability, this should have installed with base pyomo"
    )
    
from pyomo.environ import (
    Reals,
    PositiveReals,
    NonPositiveReals,
    NegativeReals,
    NonNegativeReals,
    Integers,
    PositiveIntegers,
    NonPositiveIntegers,
    NegativeIntegers,
    NonNegativeIntegers,
    Boolean,
    Binary,
    Any,
    AnyWithNone,
    EmptySet,
    UnitInterval,
    PercentFraction,
    RealInterval,
    IntegerInterval,
)

domainList = [
    Reals,
    PositiveReals,
    NonPositiveReals,
    NegativeReals,
    NonNegativeReals,
    Integers,
    PositiveIntegers,
    NonPositiveIntegers,
    NegativeIntegers,
    NonNegativeIntegers,
    Boolean,
    Binary,
    Any,
    AnyWithNone,
    EmptySet,
    UnitInterval,
    PercentFraction,
    RealInterval,
    IntegerInterval,
]


def decodeUnits(u_val):
    if isinstance(u_val, str):
        if u_val in ['', '-', 'None', ' ', 'dimensionless']:
            return pyomo_units.__getattr__('dimensionless')
        else:
            return pyomo_units.__getattr__(u_val)
    else:
        return u_val


class Formulation(ConcreteModel):
    def __init__(self):
        super(Formulation, self).__init__()
        # self._variable_counter = 1
        # self._constant_counter = 1
        self._objective_counter = 0
        self._constraint_counter = 0

        self._variable_keys = []
        self._constant_keys = []
        self._objective_keys = []
        self._runtimeObjective_keys = []
        self._objective_keys = []
        self._runtimeConstraint_keys = []
        self._constraint_keys = []
        self._allConstraint_keys = []

    def Variable(
        self, name, guess, units, description='', size=None, bounds=None, domain=None
    ):
        if domain is None:
            domain = Reals
        else:
            if domain not in domainList:
                raise RuntimeError("Invalid domain")

        if bounds is not None:
            if not isinstance(bounds, (list, tuple)):
                raise ValueError(
                    'The keyword bounds must be a 2 length list or tuple of floats'
                )
            if len(bounds) != 2:
                raise ValueError(
                    'The keyword bounds must be a 2 length list or tuple of floats'
                )
            if not isinstance(bounds[0], (float, int)):
                raise ValueError(
                    'The keyword bounds must be a 2 length list or tuple of floats'
                )
            if not isinstance(bounds[1], (float, int)):
                raise ValueError(
                    'The keyword bounds must be a 2 length list or tuple of floats'
                )
            if bounds[0] > bounds[1]:
                raise ValueError("Lower bound is higher than upper bound")

        if size is not None:
            if isinstance(size, (list, tuple)):
                for i in range(0, len(size)):
                    if not isinstance(size[i], int):
                        raise ValueError(
                            'Invalid size.  Must be an integer or list/tuple of integers'
                        )
                    if size[i] == 1 or size[i] == 0:
                        raise ValueError(
                            'A value of 0 or 1 is not valid for defining size.  Use fewer dimensions.'
                        )
                    if i == 0:
                        st = pyo.Set(initialize=list(range(0, size[i])))
                    else:
                        st *= pyo.Set(initialize=list(range(0, size[i])))
                st.construct()
                self.add_component(
                    name,
                    pyo.Var(
                        st,
                        name=name,
                        initialize=guess,
                        domain=domain,
                        bounds=bounds,
                        doc=description,
                        units=decodeUnits(units),
                    ),
                )
            else:
                if isinstance(size, int):
                    if size == 1 or size == 0:
                        self.add_component(
                            name,
                            pyo.Var(
                                name=name,
                                initialize=guess,
                                domain=domain,
                                bounds=bounds,
                                doc=description,
                                units=decodeUnits(units),
                            ),
                        )
                    else:
                        st = pyo.Set(initialize=list(range(0, size)))
                        st.construct()
                        self.add_component(
                            name,
                            pyo.Var(
                                st,
                                name=name,
                                initialize=guess,
                                domain=domain,
                                bounds=bounds,
                                doc=description,
                                units=decodeUnits(units),
                            ),
                        )
                else:
                    raise ValueError(
                        'Invalid size.  Must be an integer or list/tuple of integers'
                    )
        else:
            self.add_component(
                name,
                pyo.Var(
                    name=name,
                    initialize=guess,
                    domain=domain,
                    bounds=bounds,
                    doc=description,
                    units=decodeUnits(units),
                ),
            )
        self.__dict__[name].construct()
        self._variable_keys.append(name)
        return self.__dict__[name]

    def Constant(self, name, value, units, description='', size=None, within=None):
        if within is None:
            within = Reals
        else:
            if within not in domainList:
                raise RuntimeError("Invalid within")

        if size is not None:
            if isinstance(size, (list, tuple)):
                for i in range(0, len(size)):
                    if not isinstance(size[i], int):
                        raise ValueError(
                            'Invalid size.  Must be an integer or list/tuple of integers'
                        )
                    if size[i] == 1:
                        raise ValueError(
                            'A value of 1 is not valid for defining size.  Use fewer dimensions.'
                        )
                    if i == 0:
                        st = pyo.Set(initialize=list(range(0, size[i])))
                    else:
                        st *= pyo.Set(initialize=list(range(0, size[i])))
                st.construct()
                self.add_component(
                    name,
                    pyo.Param(
                        st,
                        name=name,
                        initialize=value,
                        within=within,
                        doc=description,
                        units=decodeUnits(units),
                        mutable=True,
                    ),
                )
            else:
                if isinstance(size, int):
                    if size == 1 or size == 0:
                        self.add_component(
                            name,
                            pyo.Param(
                                name=name,
                                initialize=value,
                                within=within,
                                doc=description,
                                units=decodeUnits(units),
                                mutable=True,
                            ),
                        )
                    else:
                        st = pyo.Set(initialize=list(range(0, size)))
                        st.construct()
                        self.add_component(
                            name,
                            pyo.Param(
                                st,
                                name=name,
                                initialize=value,
                                within=within,
                                doc=description,
                                units=decodeUnits(units),
                                mutable=True,
                            ),
                        )
                else:
                    raise ValueError(
                        'Invalid size.  Must be an integer or list/tuple of integers'
                    )
        else:
            self.add_component(
                name,
                pyo.Param(
                    name=name,
                    initialize=value,
                    within=within,
                    doc=description,
                    units=decodeUnits(units),
                    mutable=True,
                ),
            )

        self.__dict__[name].construct()
        self._constant_keys.append(name)
        return self.__dict__[name]

    def Objective(self, expr, sense=minimize):
        self._objective_counter += 1
        self.add_component(
            'objective_' + str(self._objective_counter),
            pyo.Objective(expr=expr, sense=sense),
        )
        self._objective_keys.append('objective_' + str(self._objective_counter))
        self.__dict__['objective_' + str(self._objective_counter)].construct()

    # def RuntimeObjective(self):
    #     pass

    def Constraint(self, expr):
        self._constraint_counter += 1
        conName = 'constraint_' + str(self._constraint_counter)
        self.add_component(conName, pyo.Constraint(expr=expr))
        self._constraint_keys.append(conName)
        self._allConstraint_keys.append(conName)
        self.__dict__[conName].construct()

    def RuntimeConstraint(self, outputs, operators, inputs, black_box):
        self._constraint_counter += 1
        conName = 'constraint_' + str(self._constraint_counter)
        self._runtimeConstraint_keys.append(conName)
        self._allConstraint_keys.append(conName)

        self.add_component(conName, ExternalGreyBoxBlock())
        self.__dict__[conName].construct()

        # TODO:  Need to include operators after Michael fixes things

        inputs_raw = inputs
        outputs_raw = outputs
        operators_raw = operators

        if isinstance(
            inputs_raw, (pyomo.core.base.var.IndexedVar, pyomo.core.base.var.ScalarVar)
        ):
            inputs_raw = [inputs_raw]
        elif isinstance(inputs_raw, (list, tuple)):
            inputs_raw = list(inputs_raw)
        else:
            raise ValueError("Invalid type for input variables")

        if isinstance(
            outputs_raw, (pyomo.core.base.var.IndexedVar, pyomo.core.base.var.ScalarVar)
        ):
            outputs_raw = [outputs_raw]
        elif isinstance(outputs_raw, (list, tuple)):
            outputs_raw = list(outputs_raw)
        else:
            raise ValueError("Invalid type for output variables")
        for lst in [outputs_raw, inputs_raw]:
            for vr in lst:
                if not isinstance(
                    vr, (pyomo.core.base.var.IndexedVar, pyomo.core.base.var.ScalarVar)
                ):
                    raise ValueError("Invalid type when checking inputs and outputs")

        if isinstance(operators_raw, (list, tuple)):
            operators_raw = list(operators_raw)
        elif isinstance(operators_raw, str):
            operators_raw = [operators_raw]
        else:
            raise ValueError("Invalid type for operators")
        for opr in operators_raw:
            if opr not in ["==", ">=", "<="]:
                raise ValueError("Invalid operator")

        black_box.setOptimizationVariables(inputs_raw, outputs_raw)

        outputs_raw_length = len(outputs_raw)
        operators_raw_length = len(operators_raw)

        outputs_unwrapped = []
        for ovar in outputs_raw:
            if isinstance(ovar, pyomo.core.base.var.ScalarVar):
                outputs_unwrapped.append(ovar)
            else:  # isinstance(ovar, pyomo.core.base.var.IndexedVar), validated above
                validIndices = list(ovar.index_set().data())
                for vi in validIndices:
                    outputs_unwrapped.append(ovar[vi])

        inputs_unwrapped = []
        for ivar in inputs_raw:
            if isinstance(ivar, pyomo.core.base.var.ScalarVar):
                inputs_unwrapped.append(ivar)
            else:  # isinstance(ivar, pyomo.core.base.var.IndexedVar), validated above
                validIndices = list(ivar.index_set().data())
                for vi in validIndices:
                    inputs_unwrapped.append(ivar[vi])

        black_box._NunwrappedOutputs = len(outputs_unwrapped)
        black_box._NunwrappedInputs = len(inputs_unwrapped)
        black_box.post_init_setup()

        # TODO:  Need to unwrap operators

        self.__dict__[conName].set_external_model(
            black_box, inputs=inputs_unwrapped, outputs=outputs_unwrapped
        )  # ,operators=operators_unwrapped)

    def ConstraintList(self, conList):
        for i in range(0, len(conList)):
            con = conList[i]
            if isinstance(con, (tuple, list)):
                self.RuntimeConstraint(*con)
            elif isinstance(con, dict):
                self.RuntimeConstraint(**con)
            else:
                self.Constraint(con)

    def get_variables(self):
        return [
            self.__dict__[nm]
            for nm in self.__dict__.keys()
            if nm in self._variable_keys
        ]

    def get_constants(self):
        return [
            self.__dict__[nm]
            for nm in self.__dict__.keys()
            if nm in self._constant_keys
        ]

    def get_objectives(self):
        return [
            self.__dict__[nm]
            for nm in self.__dict__.keys()
            if nm in self._objective_keys
        ]

    def get_constraints(self):
        return [
            self.__dict__[nm]
            for nm in self.__dict__.keys()
            if nm in self._allConstraint_keys
        ]

    def get_explicitConstraints(self):
        return [
            self.__dict__[nm]
            for nm in self.__dict__.keys()
            if nm in self._constraint_keys
        ]

    def get_runtimeConstraints(self):
        return [
            self.__dict__[nm]
            for nm in self.__dict__.keys()
            if nm in self._runtimeConstraint_keys
        ]

    def check_units(self):
        for i in range(1, self._objective_counter + 1):
            assert_units_consistent(self.__dict__['objective_' + str(i)])

        for i in range(1, self._constraint_counter + 1):
            if not isinstance(
                self.__dict__['constraint_' + str(i)],
                pyomo.contrib.pynumero.interfaces.external_grey_box.ExternalGreyBoxBlock,
            ):
                assert_units_consistent(self.__dict__['constraint_' + str(i)])
