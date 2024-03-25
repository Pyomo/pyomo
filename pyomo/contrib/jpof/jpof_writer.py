#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

__all__ = ['ProblemWriter_json']

#
# JSON Problem Writer Plugin
#
"""
The goal of this format is to enable rapid serialization of Pyomo models
capturing information about the mutable parameters and fixed variables
in the model.  Additionally, this format uses JSON to enable robust
parsing with standard techniques.

Note that this format does not do some of the reformulation that is
done with serialization formats like LP or NL, where linear or quadratic 
structure is identified.  Instead, the assumption is that the solver
interface would analyze the model expressions in a manner that is best
suited for their analysis.

This formulation expresses objective and constraint expressions in a
compact expression tree.  However, this uses a comma-separated format
that is relatively easy to read.
"""

try:
    basestring
except:
    basestring = str

import itertools
import logging
import operator
import os
import time
try:
    import ujson as json
except:
    import json

from math import isclose

from pyomo.common.gc_manager import PauseGC
from pyomo.opt import ProblemFormat, AbstractProblemWriter, WriterFactory
from pyomo.core.expr import current as EXPR
from pyomo.core.expr.numvalue import (NumericConstant,
                                      native_numeric_types,
                                      value,
                                      is_constant,
                                      is_fixed)
from pyomo.core.base import SymbolMap, NameLabeler, _ExpressionData, SortComponents, var, param, Var, ExternalFunction, ComponentMap, Objective, Constraint, SOSConstraint, Suffix, Param
import pyomo.core.base.suffix
from pyomo.repn.standard_repn import generate_standard_repn

import pyomo.core.kernel.suffix
from pyomo.core.kernel.block import IBlock
from pyomo.core.kernel.expression import IIdentityExpression
from pyomo.core.kernel.variable import IVariable

from six import iteritems
from six.moves import xrange

logger = logging.getLogger('pyomo.core')



class JPOFWriter(AbstractProblemWriter):

    def __init__(self):
        pass

    def __call__(self,
                 model,
                 filename=None,
                 solver_capability=None,
                 io_options={},
                 sets=[]):

        # Make sure not to modify the user's dictionary, they may be
        # reusing it outside of this call
        io_options = dict(io_options)

        # Skip writing constraints whose body section is fixed (i.e., no variables)
        skip_trivial_constraints = io_options.pop("skip_trivial_constraints", False)

        # How much effort do we want to put into ensuring the
        # file is written deterministically for a Pyomo model:
        #    0 : None
        #    1 : sort keys of indexed components (default)
        #    2 : sort keys AND sort names (over declaration order)
        file_determinism = io_options.pop("file_determinism", 1)

        # Add row/col data that labels variable and constraint indices in the NLP
        # matrix.
        self._symbolic_solver_labels = io_options.pop("symbolic_solver_labels", False)

        # If False, we raise an error if fixed variables are
        # encountered in the list of active variables (i.e., an
        # expression containing some fixed variable was not
        # preprocessed after the variable was fixed). If True, we
        # allow this case and modify the variable bounds section to
        # fix the variable.
        self._output_fixed_variable_bounds = io_options.pop("output_fixed_variable_bounds", False)

        # If False, unused variables will not be included in
        # the JSON file. Otherwise, include all variables in
        # the bounds sections.
        include_all_variable_bounds = io_options.pop("include_all_variable_bounds", False)

        if len(io_options):
            raise ValueError(
                "ProblemWriter_json passed unrecognized io_options:\n\t" +
                "\n\t".join("%s = %s" % (k,v) for k,v in iteritems(io_options)))

        if filename is None:
            filename = model.name + ".json"


        # Setup labeler
        self._name_labeler = NameLabeler()

        # Pause the GC for the duration of this method
        with PauseGC() as pgc:
            symbol_map, model_dict = self._collect_model(
                model,
                solver_capability,
                skip_trivial_constraints=skip_trivial_constraints,
                file_determinism=file_determinism,
                include_all_variable_bounds=include_all_variable_bounds,
                sets=sets)
        with open(filename,"w") as f:
            json.dump(model_dict, f, indent=2)

        # Cleanup memory
        self._symbolic_solver_labels = False
        self._output_fixed_variable_bounds = False
        self._name_labeler = None

        return filename, symbol_map

    def _nonlinear_expr_to_string(self, exp, used_vars, used_params):
        slist = []
        self._nonlinear_expr_to_list(exp, slist, used_vars, used_params)
        return ",".join(slist)

    def _nonlinear_expr_to_list(self, exp, slist, used_vars, used_params):
        exp_type = type(exp)

        if exp_type is list:
            raise RuntimeError("LIST EXPRESSION???")
            # this is an implied summation of expressions (we did not
            # create a new sum expression for efficiency) this should
            # be a list of tuples where [0] is the coeff and [1] is
            # the expr to write
            nary_sum_str, binary_sum_str, coef_term_str = \
                self._op_string[EXPR.SumExpressionBase]
            n = len(exp)
            if n > 2:
                OUTPUT.write(nary_sum_str % (n))
                for i in xrange(0,n):
                    assert(exp[i].__class__ is tuple)
                    coef = exp[i][0]
                    child_exp = exp[i][1]
                    if coef != 1:
                        OUTPUT.write(coef_term_str % (coef))
                    self._print_nonlinear_terms_NL(child_exp)
            else: # n == 1 or 2
                for i in xrange(0,n):
                    assert(exp[i].__class__ is tuple)
                    coef = exp[i][0]
                    child_exp = exp[i][1]
                    if i != n-1:
                        # need the + op if it is not the last entry in the list
                        OUTPUT.write(binary_sum_str)
                    if coef != 1:
                        OUTPUT.write(coef_term_str % (coef))
                    self._print_nonlinear_terms_NL(child_exp)

        elif exp_type in native_numeric_types:
            slist.extend(("N", str(exp)))

        elif exp.is_expression_type():
            #
            # Identify NPV expressions
            #
            if exp.is_constant():
                slist.extend(("N", str(value(exp))))
            #
            # We are assuming that _Constant_* expression objects
            # have been preprocessed to form constant values.
            #
            elif exp.__class__ in [EXPR.SumExpression,EXPR.NPV_SumExpression]:
                n = exp.nargs()
                const = 0
                vargs = []
                for v in exp.args:
                    if v.__class__ in native_numeric_types:
                        const += v
                    else:
                        vargs.append(v)
                if not isclose(const, 0.0):
                    vargs.append(const)
                n = len(vargs)
                if n == 2:
                    slist.append("+")
                    self._nonlinear_expr_to_list(vargs[0], slist, used_vars, used_params)
                    self._nonlinear_expr_to_list(vargs[1], slist, used_vars, used_params)
                elif n == 1:
                    self._nonlinear_expr_to_list(vargs[0], slist, used_vars, used_params)
                else:
                    slist.extend(("sum",str(n)))
                    for child_exp in vargs:
                        self._nonlinear_expr_to_list(child_exp, slist, used_vars, used_params)

            elif exp_type is EXPR.SumExpressionBase:
                slist.append("+")
                self._nonlinear_expr_to_list(exp.arg(0), slist, used_vars, used_params)
                self._nonlinear_expr_to_list(exp.arg(1), slist, used_vars, used_params)

            elif exp_type is EXPR.MonomialTermExpression:
                slist.append("*")
                if not exp.is_potentially_variable():
                    self._nonlinear_expr_to_list(value(exp.arg(0)), slist, used_vars, used_params)
                else:
                    self._nonlinear_expr_to_list(exp.arg(0), slist, used_vars, used_params)
                self._nonlinear_expr_to_list(exp.arg(1), slist, used_vars, used_params)

            elif exp_type in [EXPR.ProductExpression, EXPR.NPV_ProductExpression]:
                slist.append("*")
                self._nonlinear_expr_to_list(exp.arg(0), slist, used_vars, used_params)
                self._nonlinear_expr_to_list(exp.arg(1), slist, used_vars, used_params)

            elif exp_type in [EXPR.DivisionExpression, EXPR.NPV_DivisionExpression]:
                assert exp.nargs() == 2
                slist.append("/")
                self._nonlinear_expr_to_list(exp.arg(0), slist, used_vars, used_params)
                self._nonlinear_expr_to_list(exp.arg(1), slist, used_vars, used_params)

            # elif exp_type in [EXPR.ReciprocalExpression, EXPR.NPV_ReciprocalExpression]:
            #     assert exp.nargs() == 1
            #     slist.append("/")
            #     self._nonlinear_expr_to_list(1.0, slist, used_vars, used_params)
            #     self._nonlinear_expr_to_list(exp.arg(0), slist, used_vars, used_params)

            elif exp_type in [EXPR.NegationExpression, EXPR.NPV_NegationExpression]:
                assert exp.nargs() == 1
                slist.append("neg")
                self._nonlinear_expr_to_list(exp.arg(0), slist, used_vars, used_params)

            elif exp_type in [EXPR.PowExpression, EXPR.NPV_PowExpression]:
                assert exp.nargs() == 2
                slist.append("pow")
                self._nonlinear_expr_to_list(exp.arg(0), slist, used_vars, used_params)
                self._nonlinear_expr_to_list(exp.arg(1), slist, used_vars, used_params)

            elif isinstance(exp, EXPR.UnaryFunctionExpression) or isinstance(exp, EXPR.NPV_UnaryFunctionExpression):
                assert exp.nargs() == 1
                slist.append(exp.name)
                self._nonlinear_expr_to_list(exp.arg(0), slist, used_vars, used_params)

            elif exp_type is EXPR.Expr_ifExpression:
                pass
                #slist.append("if")
                #self._nonlinear_expr_to_list(exp._if, slist)
                #self._nonlinear_expr_to_list(exp._then, slist)
                #self._nonlinear_expr_to_list(exp._else, slist)

            elif exp_type is EXPR.InequalityExpression:
                pass
                #if exp._strict:
                #    slist.append("<")
                #else:
                #    slist.append("<=")
                #self._nonlinear_expr_to_list(exp.arg(0), slist)
                #self._nonlinear_expr_to_list(exp.arg(1), slist)

            elif exp_type is EXPR.RangedExpression:
                pass
                #left = exp.arg(0)
                #middle = exp.arg(1)
                #right = exp.arg(2)
                #OUTPUT.write(and_str)
                #if exp._strict[0]:
                #    OUTPUT.write(lt_str)
                #else:
                #    OUTPUT.write(le_str)
                #self._print_nonlinear_terms_NL(left)
                #self._print_nonlinear_terms_NL(middle)
                #if exp._strict[1]:
                #    OUTPUT.write(lt_str)
                #else:
                #    OUTPUT.write(le_str)
                #self._print_nonlinear_terms_NL(middle)
                #self._print_nonlinear_terms_NL(right)

            elif exp_type is EXPR.EqualityExpression:
                pass
                #OUTPUT.write(self._op_string[EXPR.EqualityExpression])
                #self._print_nonlinear_terms_NL(exp.arg(0))
                #self._print_nonlinear_terms_NL(exp.arg(1))

            elif isinstance(exp, (_ExpressionData, IIdentityExpression)) \
                 or isinstance(exp, (_ExpressionData, NPV_IIdentityExpression)):
                self._nonlinear_expr_to_list(exp.arg(0), slist, used_vars, used_params)

            else:
                raise ValueError(
                    "Unsupported expression type (%s) in _nonlinear_expr_to_list"
                    % (exp_type))

        elif isinstance(exp, (var._VarData, IVariable)):
            slist.extend(("V", str(self._varID[id(exp)])))
            used_vars.add(id(exp))

        elif isinstance(exp,param._ParamData):
            slist.extend(("P", str(self._paramID[id(exp)])))
            used_params.add(id(exp))

        elif isinstance(exp,NumericConstant):
            slist.extend(("N", str(value(exp))))

        else:
            raise ValueError(
                "Unsupported expression type (%s) in _nonlinear_expr_to_list"
                % (exp_type))

    def _collect_model(self, model,
                        solver_capability,
                        skip_trivial_constraints=False,
                        file_determinism=1,
                        include_all_variable_bounds=False,
                        sets=[]):

        sorter = SortComponents.unsorted
        if file_determinism >= 1:
            sorter = sorter | SortComponents.indices
            if file_determinism >= 2:
                sorter = sorter | SortComponents.alphabetical

        # Cache the list of model blocks so we don't have to call
        # model.block_data_objects() many many times
        all_blocks_list = list(model.block_data_objects(active=True, sort=sorter))

        # create a deterministic var labeling
        Vars_dict = dict( enumerate( model.component_data_objects(Var, sort=sorter) ) )
        self._varID = dict((id(val),key) for key,val in iteritems(Vars_dict))
        Params_dict = dict( enumerate( model.component_data_objects(Param, sort=sorter) ) )
        self._paramID = dict((id(val),key) for key,val in iteritems(Params_dict))
        used_vars = set()
        used_params = set()

        output_fixed_variable_bounds = self._output_fixed_variable_bounds
        symbolic_solver_labels = self._symbolic_solver_labels

        # create the symbol_map
        symbol_map = SymbolMap()
        name_labeler = self._name_labeler

        indexed_vars = set()
        indexed_params = set()

        data = {}
        data['__metadata__'] = {'version':20210301, 'format':'JSON Parameterized Optimization Format (JPOF)'}

        Model = {}
        data['model'] = Model
        Model['config'] = {}
        Model['config']['symbolic_solver_labels'] = int(symbolic_solver_labels)
        Model['config']['skip_trivial_constraints'] = int(skip_trivial_constraints)
        Model['config']['file_determinism'] = file_determinism
        Model['config']['include_all_variable_bounds'] = int(include_all_variable_bounds)
        Model['obj'] = []
        Model['con'] = []

        #
        # Count number of objectives and build the repns
        #
        n_obj = 0
        n_con = 0
        for block in all_blocks_list:

            for active_objective in block.component_data_objects(Objective,
                                                                 active=True,
                                                                 sort=sorter,
                                                                 descend_into=False):
                objdata = {}
                objdata['expr'] = self._nonlinear_expr_to_string(active_objective, used_vars, used_params)
                if active_objective.is_minimizing():
                    objdata['sense'] = 'min'
                else:
                    objdata['sense'] = 'max'
                if symbolic_solver_labels:
                    objdata['label'] = name_labeler(active_objective)

                symbol_map.addSymbols([(active_objective, "o%d"%n_obj)])
                symbol_map.alias(symbol_map.bySymbol["o0"](),"__default_objective__")
                n_obj += 1
                Model['obj'].append(objdata)

            for constraint_data in block.component_data_objects(Constraint,
                                                                active=True,
                                                                sort=sorter,
                                                                descend_into=False):
                if (not constraint_data.has_lb()) and \
                   (not constraint_data.has_ub()):
                    assert not constraint_data.equality
                    continue  # non-binding, so skip

                _type = getattr(constraint_data, '_complementarity', None)
                if not _type is None:
                    # Skip complementarity conditions for now
                    continue

                condata = {}
                condata['expr'] = self._nonlinear_expr_to_string(constraint_data.body, used_vars, used_params)
                if symbolic_solver_labels:
                    condata['label'] = name_labeler(constraint_data)

                L = None
                U = None
                if constraint_data.has_lb():
                    L = self._get_bound(constraint_data.lower, used_vars, used_params)
                else:
                    assert constraint_data.has_ub()
                if constraint_data.has_ub():
                    U = self._get_bound(constraint_data.upper, used_vars, used_params)
                else:
                    assert constraint_data.has_lb()
                if constraint_data.equality:
                    assert L == U
                    if L is not None:
                        condata['eq'] = L
                else:
                    if L is not None:
                        condata['geq'] = L
                    if U is not None:
                        condata['leq'] = U

                n_con += 1
                Model['con'].append(condata)

        Model['var'] = []
        v = Model['var']
        if file_determinism >= 1:
            varids = sorted(self._varID[i] for i in used_vars)
        else:
            varids = set(self._varID[i] for i in used_vars)
        for i in varids:
            tmp = {}
            v_ = Vars_dict[i]
            if symbolic_solver_labels:
                tmp['label'] = str(v_)
                if v_.parent_component().is_indexed():
                    indexed_vars.add(v_.parent_component().name)
            tmp['id'] = i
            if v_.value is not None:
                tmp['value'] = v_.value
            lb = self._get_bound(v_.lb)
            if lb is not None:
                tmp['lb'] = lb
            ub = self._get_bound(v_.ub)
            if ub is not None:
                tmp['ub'] = ub
            if v_.is_binary():
                tmp['type'] = 'B'
            elif v_.is_integer():
                tmp['type'] = 'Z'
            else:
                tmp['type'] = 'R'
            tmp['fixed'] = int(v_.fixed)
            v.append(tmp)

        Model['param'] = []
        p = Model['param']
        if file_determinism >= 1:
            paramids = sorted(self._paramID[i] for i in used_params)
        else:
            paramids = set(self._paramID[i] for i in used_params)
        for i in paramids:
            tmp = {}
            p_ = Params_dict[i]
            if symbolic_solver_labels:
                tmp['label'] = str(p_)
                if p_.parent_component().is_indexed():
                    indexed_params.add(p_.parent_component().name)
            tmp['id'] = i
            if p_.value is not None:
                tmp['value'] = p_.value
            p.append(tmp)

        Model['set'] = {}
        s = Model['set']
        for setname in sets:
            modelset = getattr(model,setname)
            values = [str(value) for value in modelset]
            s[setname] = values

        Model['indexed_vars'] = list(sorted(indexed_vars))
        Model['indexed_params'] = list(sorted(indexed_params))

        return symbol_map, data

    def _get_bound(self, exp, used_vars=None, used_params=None):
        if exp is None:
            return None
        if is_constant(exp):
            return value(exp)
        if not is_fixed(exp):
            raise ValueError("non-fixed bound or weight: " + str(exp))
        if used_vars is None or used_params is None:
            return self._nonlinear_expr_to_string(exp, set(), set())
        return self._nonlinear_expr_to_string(exp, used_vars, used_params)

