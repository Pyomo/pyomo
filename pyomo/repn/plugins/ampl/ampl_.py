#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

#
# AMPL Problem Writer Plugin
#

__all__ = ['ProblemWriter_nl']

try:
    basestring
except:
    basestring = str

import itertools
import logging
import operator
import os
import time

from pyutilib.math.util import isclose
from pyutilib.misc import PauseGC

from pyomo.opt import ProblemFormat
from pyomo.opt.base import *
from pyomo.core.expr import current as EXPR
from pyomo.core.expr.numvalue import (NumericConstant,
                                      native_numeric_types,
                                      value)
from pyomo.core.base import *
from pyomo.core.base import SymbolMap, Block
from pyomo.core.base.var import Var
from pyomo.core.base import _ExpressionData, Expression, SortComponents
from pyomo.core.base import var
from pyomo.core.base import param
import pyomo.core.base.suffix
from pyomo.repn.standard_repn import StandardRepn, generate_standard_repn

import pyomo.core.kernel.suffix
from pyomo.core.kernel.block import IBlock
from pyomo.core.kernel.expression import IIdentityExpression
from pyomo.core.kernel.variable import IVariable

from six import itervalues, iteritems
from six.moves import xrange, zip

logger = logging.getLogger('pyomo.core')

_intrinsic_function_operators = {
    'log':    'o43',
    'log10':  'o42',
    'sin':    'o41',
    'cos':    'o46',
    'tan':    'o38',
    'sinh':   'o40',
    'cosh':   'o45',
    'tanh':   'o37',
    'asin':   'o51',
    'acos':   'o53',
    'atan':   'o49',
    'exp':    'o44',
    'sqrt':   'o39',
    'asinh':  'o50',
    'acosh':  'o52',
    'atanh':  'o47',
    'pow':    'o5',
    'abs':    'o15',
    'ceil':   'o13',
    'floor':  'o14'
}

# build string templates
def _build_op_template():
    _op_template = {}
    _op_comment = {}

    prod_template = "o2{C}\n"
    prod_comment = "\t#*"
    div_template = "o3{C}\n"
    div_comment = "\t#/"
    _op_template[EXPR.ProductExpression] = prod_template
    _op_comment[EXPR.ProductExpression] = prod_comment
    _op_template[EXPR.ReciprocalExpression] = div_template
    _op_comment[EXPR.ReciprocalExpression] = div_comment
    del prod_template
    del prod_comment
    del div_template
    del div_comment

    _op_template[EXPR.ExternalFunctionExpression] = ("f%d %d{C}\n", #function
                                                      "h%d:%s{C}\n") #string arg
    _op_comment[EXPR.ExternalFunctionExpression] = ("\t#%s", #function
                                                     "")      #string arg

    for opname in _intrinsic_function_operators:
        _op_template[opname] = _intrinsic_function_operators[opname]+"{C}\n"
        _op_comment[opname] = "\t#"+opname

    _op_template[EXPR.Expr_ifExpression] = "o35{C}\n"
    _op_comment[EXPR.Expr_ifExpression] = "\t#if"

    _op_template[EXPR.InequalityExpression] = ("o21{C}\n", # and
                                                "o22{C}\n", # <
                                                "o23{C}\n") # <=
    _op_comment[EXPR.InequalityExpression] = ("\t#and", # and
                                               "\t#lt",  # <
                                               "\t#le")  # <=

    _op_template[EXPR.EqualityExpression] = "o24{C}\n"
    _op_comment[EXPR.EqualityExpression] = "\t#eq"

    _op_template[var._VarData] = "v%d{C}\n"
    _op_comment[var._VarData] = "\t#%s"

    _op_template[param._ParamData] = "n%r{C}\n"
    _op_comment[param._ParamData] = ""

    _op_template[NumericConstant] = "n%r{C}\n"
    _op_comment[NumericConstant] = ""

    _op_template[EXPR.SumExpressionBase] = (
        "o54{C}\n%d\n", # nary +
        "o0{C}\n",      # +
        "o2\n" + _op_template[NumericConstant] ) # * coef
    _op_comment[EXPR.SumExpressionBase] = ("\t#sumlist", # nary +
                                        "\t#+",       # +
                                        _op_comment[NumericConstant]) # * coef
    _op_template[EXPR.NegationExpression] = "o16{C}\n"
    _op_comment[EXPR.NegationExpression] = "\t#-"

    return _op_template, _op_comment



def _get_bound(exp):
    if exp is None:
        return None
    if is_fixed(exp):
        return value(exp)
    raise ValueError("non-fixed bound or weight: " + str(exp))

class StopWatch(object):

    def __init__(self):
        self.start = time.time()

    def report(self, msg):
        print(msg+" (seconds): "+str(time.time()-self.start))

    def reset(self):
        self.start = time.time()

class _Counter(object):

    def __init__(self, start):
        self._id = start

    def __call__(self, obj):
        tmp = self._id
        self._id += 1
        return tmp

class ModelSOS(object):

    class AmplSuffix(object):

        def __init__(self,name):
            self.name = name
            self.ids = []
            self.vals = []

        def add(self,idx,val):
            if idx in self.ids:
                raise RuntimeError(
                    "The NL file format does not support multiple nonzero "
                    "values for a single component and suffix. \n"
                    "Suffix Name:  %s\n"
                    "Component ID: %s\n" % (self.name, idx))
            else:
                self.ids.append(idx)
                self.vals.append(val)

        def genfilelines(self):
            base_line = "{0} {1}\n"
            return [base_line.format(idx, val)
                    for idx, val in zip(self.ids,self.vals) if val != 0]

        def is_empty(self):
            return not bool(len(self.ids))

    def __init__(self,ampl_var_id, varID_map):

        self.ampl_var_id = ampl_var_id
        self.sosno = self.AmplSuffix('sosno')
        self.ref = self.AmplSuffix('ref')
        self.block_cntr = 0
        self.varID_map = varID_map

    def count_constraint(self,soscondata):

        ampl_var_id = self.ampl_var_id
        varID_map = self.varID_map

        if hasattr(soscondata, 'get_items'):
            sos_items = list(soscondata.get_items())
        else:
            sos_items = list(soscondata.items())

        if len(sos_items) == 0:
            return

        level = soscondata.level

        # Identifies the the current set from others in the NL file
        self.block_cntr += 1

        # If SOS1, the identifier must be positive
        # if SOS2, the identifier must be negative
        sign_tag = None
        if level == 1:
            sign_tag = 1
        elif level == 2:
            sign_tag = -1
        else:
            raise ValueError("SOSContraint '%s' has sos type='%s', "
                             "which is not supported by the NL file interface" \
                                 % (soscondata.name, level))

        for vardata, weight in sos_items:
            weight = _get_bound(weight)
            if weight < 0:
                raise ValueError(
                    "Cannot use negative weight %f "
                    "for variable %s is special ordered "
                    "set %s " % (weight, vardata.name, soscondata.name))
            if vardata.fixed:
                raise ValueError(
                    "SOSConstraint '%s' includes a fixed Variable '%s'. "
                    "This is currently not supported. Deactivate this constraint "
                    "in order to proceed"
                    % (soscondata.name, vardata.name))

            ID = ampl_var_id[varID_map[id(vardata)]]
            self.sosno.add(ID,self.block_cntr*sign_tag)
            self.ref.add(ID,weight)

class RepnWrapper(object):

    __slots__ = ('repn','linear_vars','nonlinear_vars')

    def __init__(self,repn,linear,nonlinear):
        self.repn = repn
        self.linear_vars = linear
        self.nonlinear_vars = nonlinear


@WriterFactory.register('nl', 'Generate the corresponding AMPL NL file.')
class ProblemWriter_nl(AbstractProblemWriter):


    def __init__(self):
        AbstractProblemWriter.__init__(self, ProblemFormat.nl)
        self._ampl_var_id = {}
        self._ampl_con_id = {}
        self._ampl_obj_id = {}
        self._OUTPUT = None
        self._varID_map = None

    def __call__(self,
                 model,
                 filename,
                 solver_capability,
                 io_options):

        # Rebuild the OP template (as the expression tree system may
        # have been switched)
        _op_template, _op_comment = _build_op_template()

        # Make sure not to modify the user's dictionary, they may be
        # reusing it outside of this call
        io_options = dict(io_options)

        # NOTE: io_options is a simple dictionary of keyword-value pairs
        #       specific to this writer. that said, we are not good
        #       about enforcing consistency between the io_options and
        #       kwds - arguably, we should go with the former or the
        #       latter, but not support both. then again, the kwds are
        #       not used anywhere within this function.

        # Print timing after writing each section of the NL file
        show_section_timing = io_options.pop("show_section_timing", False)

        # Skip writing constraints whose body section is fixed (i.e., no variables)
        skip_trivial_constraints = io_options.pop("skip_trivial_constraints", False)

        # How much effort do we want to put into ensuring the
        # NL file is written deterministically for a Pyomo model:
        #    0 : None
        #    1 : sort keys of indexed components (default)
        #    2 : sort keys AND sort names (over declaration order)
        file_determinism = io_options.pop("file_determinism", 1)

        # Write the corresponding .row and .col files for the NL files
        # identifying variable and constraint indices in the NLP
        # matrix.
        symbolic_solver_labels = io_options.pop("symbolic_solver_labels", False)

        # If False, we raise an error if fixed variables are
        # encountered in the list of active variables (i.e., an
        # expression containing some fixed variable was not
        # preprocessed after the variable was fixed). If True, we
        # allow this case and modify the variable bounds section to
        # fix the variable.
        output_fixed_variable_bounds = \
                io_options.pop("output_fixed_variable_bounds", False)

        # If False, unused variables will not be included in
        # the NL file. Otherwise, include all variables in
        # the bounds sections.
        include_all_variable_bounds = \
            io_options.pop("include_all_variable_bounds", False)

        if len(io_options):
            raise ValueError(
                "ProblemWriter_nl passed unrecognized io_options:\n\t" +
                "\n\t".join("%s = %s" % (k,v) for k,v in iteritems(io_options)))

        if filename is None:
            filename = model.name + ".nl"

        # Generate the operator strings templates. The value of
        # symbolic_solver_labels determines whether or not to
        # include "nl comments" (the equivalent AMPL functionality
        # is "option nl_comments 1").
        self._op_string = {}
        for optype in _op_template:
            template_str = _op_template[optype]
            comment_str = _op_comment[optype]
            if type(template_str) is tuple:
                op_strings = []
                for i in xrange(len(template_str)):
                    if symbolic_solver_labels:
                        op_strings.append(template_str[i].format(C=comment_str[i]))
                    else:
                        op_strings.append(template_str[i].format(C=""))
                self._op_string[optype] = tuple(op_strings)
            else:
                if symbolic_solver_labels:
                    self._op_string[optype] = template_str.format(C=comment_str)
                else:
                    self._op_string[optype] = template_str.format(C="")

        # making these attributes so they do not need to be
        # passed into _print_nonlinear_terms_NL
        self._symbolic_solver_labels = symbolic_solver_labels
        self._output_fixed_variable_bounds = output_fixed_variable_bounds
        # Speeds up calling name on every component when
        # writing .row and .col files (when symbolic_solver_labels is True)
        self._name_labeler = NameLabeler()

        # Pause the GC for the duration of this method
        with PauseGC() as pgc:
            with open(filename,"w") as f:
                self._OUTPUT = f
                symbol_map = self._print_model_NL(
                    model,
                    solver_capability,
                    show_section_timing=show_section_timing,
                    skip_trivial_constraints=skip_trivial_constraints,
                    file_determinism=file_determinism,
                    include_all_variable_bounds=include_all_variable_bounds)

        self._symbolic_solver_labels = False
        self._output_fixed_variable_bounds = False
        self._name_labeler = None

        self._OUTPUT = None
        self._varID_map = None
        self._op_string = None
        return filename, symbol_map

    def _print_quad_term(self, v1, v2):
        OUTPUT = self._OUTPUT
        if v1 is not v2:
            prod_str = self._op_string[EXPR.ProductExpression]
            OUTPUT.write(prod_str)
            self._print_nonlinear_terms_NL(v1)
            self._print_nonlinear_terms_NL(v2)
        else:
            intr_expr_str = self._op_string['pow']
            OUTPUT.write(intr_expr_str)
            self._print_nonlinear_terms_NL(v1)
            OUTPUT.write(self._op_string[NumericConstant] % (2))

    def _print_standard_quadratic_NL(self,
                                     quadratic_vars,
                                     quadratic_coefs):
        OUTPUT = self._OUTPUT
        nary_sum_str, binary_sum_str, coef_term_str = \
            self._op_string[EXPR.SumExpressionBase]
        assert len(quadratic_vars) == len(quadratic_coefs)
        if len(quadratic_vars) == 1:
            pass
        else:
            if len(quadratic_vars) == 2:
                OUTPUT.write(binary_sum_str)
            else:
                assert len(quadratic_vars) > 2
                OUTPUT.write(nary_sum_str % (len(quadratic_vars)))
            # now we need to do a sort to ensure deterministic output
            # as the compiled quadratic representation does not preserve
            # any ordering
            old_quadratic_vars = quadratic_vars
            old_quadratic_coefs = quadratic_coefs
            self_varID_map = self._varID_map
            quadratic_vars = []
            quadratic_coefs = []
            for (i, (v1, v2)) in sorted(enumerate(old_quadratic_vars),
                                        key=lambda x: (self_varID_map[id(x[1][0])],
                                                       self_varID_map[id(x[1][1])])):
                quadratic_coefs.append(old_quadratic_coefs[i])
                if self_varID_map[id(v1)] <= self_varID_map[id(v2)]:
                    quadratic_vars.append((v1,v2))
                else:
                    quadratic_vars.append((v2,v1))
        for i in range(len(quadratic_vars)):
            coef = quadratic_coefs[i]
            v1, v2 = quadratic_vars[i]
            if coef != 1:
                OUTPUT.write(coef_term_str % (coef))
            self._print_quad_term(v1, v2)

    def _print_nonlinear_terms_NL(self, exp):
        OUTPUT = self._OUTPUT
        exp_type = type(exp)
        # JDS: check list first so that after this, we know that exp
        # must be some form of NumericValue
        if exp_type is list:
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
            else:
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
            OUTPUT.write(self._op_string[NumericConstant]
                         % (exp))

        elif exp.is_expression_type():
            #
            # Identify NPV expressions
            #
            if not exp.is_potentially_variable():
                OUTPUT.write(self._op_string[NumericConstant] % (value(exp)))
            #
            # We are assuming that _Constant_* expression objects
            # have been preprocessed to form constant values.
            #
            elif exp.__class__ is EXPR.SumExpression:
                nary_sum_str, binary_sum_str, coef_term_str = \
                    self._op_string[EXPR.SumExpressionBase]
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
                    OUTPUT.write(binary_sum_str)
                    self._print_nonlinear_terms_NL(vargs[0])
                    self._print_nonlinear_terms_NL(vargs[1])
                else:
                    OUTPUT.write(nary_sum_str % (n))
                    for child_exp in vargs:
                        self._print_nonlinear_terms_NL(child_exp)

            elif exp_type is EXPR.SumExpressionBase:
                nary_sum_str, binary_sum_str, coef_term_str = \
                    self._op_string[EXPR.SumExpressionBase]
                OUTPUT.write(binary_sum_str)
                self._print_nonlinear_terms_NL(exp.arg(0))
                self._print_nonlinear_terms_NL(exp.arg(1))

            elif exp_type is EXPR.MonomialTermExpression:
                prod_str = self._op_string[EXPR.ProductExpression]
                OUTPUT.write(prod_str)
                self._print_nonlinear_terms_NL(value(exp.arg(0)))
                self._print_nonlinear_terms_NL(exp.arg(1))

            elif exp_type is EXPR.ProductExpression:
                prod_str = self._op_string[EXPR.ProductExpression]
                OUTPUT.write(prod_str)
                self._print_nonlinear_terms_NL(exp.arg(0))
                self._print_nonlinear_terms_NL(exp.arg(1))

            elif exp_type is EXPR.ReciprocalExpression:
                assert exp.nargs() == 1
                div_str = self._op_string[EXPR.ReciprocalExpression]
                OUTPUT.write(div_str)
                self._print_nonlinear_terms_NL(1.0)
                self._print_nonlinear_terms_NL(exp.arg(0))

            elif exp_type is EXPR.NegationExpression:
                assert exp.nargs() == 1
                OUTPUT.write(self._op_string[EXPR.NegationExpression])
                self._print_nonlinear_terms_NL(exp.arg(0))

            elif exp_type is EXPR.ExternalFunctionExpression:
                # We have found models where external functions with
                # strictly fixed/constant arguments causes AMPL to
                # SEGFAULT.  To be safe, we will collapse fixed
                # arguments to scalars and if the entire expression is
                # constant, we will eliminate the external function
                # call entirely.
                if exp.is_fixed():
                    self._print_nonlinear_terms_NL(exp())
                    return
                fun_str, string_arg_str = \
                    self._op_string[EXPR.ExternalFunctionExpression]
                if not self._symbolic_solver_labels:
                    OUTPUT.write(fun_str
                                 % (self.external_byFcn[exp._fcn._function][1],
                                    exp.nargs()))
                else:
                    # Note: exp.name fails
                    OUTPUT.write(fun_str
                                 % (self.external_byFcn[exp._fcn._function][1],
                                    exp.nargs(),
                                    exp.name))
                for arg in exp.args:
                    if isinstance(arg, basestring):
                        OUTPUT.write(string_arg_str % (len(arg), arg))
                    elif arg.is_fixed():
                        self._print_nonlinear_terms_NL(arg())
                    else:
                        self._print_nonlinear_terms_NL(arg)

            elif exp_type is EXPR.PowExpression:
                intr_expr_str = self._op_string['pow']
                OUTPUT.write(intr_expr_str)
                self._print_nonlinear_terms_NL(exp.arg(0))
                self._print_nonlinear_terms_NL(exp.arg(1))

            elif isinstance(exp, EXPR.UnaryFunctionExpression):
                assert exp.nargs() == 1
                intr_expr_str = self._op_string.get(exp.name)
                if intr_expr_str is not None:
                    OUTPUT.write(intr_expr_str)
                else:
                    logger.error("Unsupported unary function ({0})".format(exp.name))
                    raise TypeError("ASL writer does not support '%s' expressions"
                                    % (exp.name))
                self._print_nonlinear_terms_NL(exp.arg(0))

            elif exp_type is EXPR.Expr_ifExpression:
                OUTPUT.write(self._op_string[EXPR.Expr_ifExpression])
                self._print_nonlinear_terms_NL(exp._if)
                self._print_nonlinear_terms_NL(exp._then)
                self._print_nonlinear_terms_NL(exp._else)

            elif exp_type is EXPR.InequalityExpression:
                and_str, lt_str, le_str = \
                    self._op_string[EXPR.InequalityExpression]
                left = exp.arg(0)
                right = exp.arg(1)
                if exp._strict:
                    OUTPUT.write(lt_str)
                else:
                    OUTPUT.write(le_str)
                self._print_nonlinear_terms_NL(left)
                self._print_nonlinear_terms_NL(right)

            elif exp_type is EXPR.RangedExpression:
                and_str, lt_str, le_str = \
                    self._op_string[EXPR.InequalityExpression]
                left = exp.arg(0)
                middle = exp.arg(1)
                right = exp.arg(2)
                OUTPUT.write(and_str)
                if exp._strict[0]:
                    OUTPUT.write(lt_str)
                else:
                    OUTPUT.write(le_str)
                self._print_nonlinear_terms_NL(left)
                self._print_nonlinear_terms_NL(middle)
                if exp._strict[1]:
                    OUTPUT.write(lt_str)
                else:
                    OUTPUT.write(le_str)
                self._print_nonlinear_terms_NL(middle)
                self._print_nonlinear_terms_NL(right)

            elif exp_type is EXPR.EqualityExpression:
                OUTPUT.write(self._op_string[EXPR.EqualityExpression])
                self._print_nonlinear_terms_NL(exp.arg(0))
                self._print_nonlinear_terms_NL(exp.arg(1))

            elif isinstance(exp, (_ExpressionData, IIdentityExpression)):
                self._print_nonlinear_terms_NL(exp.expr)

            else:
                raise ValueError(
                    "Unsupported expression type (%s) in _print_nonlinear_terms_NL"
                    % (exp_type))

        elif isinstance(exp, (var._VarData, IVariable)) and \
             (not exp.is_fixed()):
            #(self._output_fixed_variable_bounds or
            if not self._symbolic_solver_labels:
                OUTPUT.write(self._op_string[var._VarData]
                             % (self.ampl_var_id[self._varID_map[id(exp)]]))
            else:
                OUTPUT.write(self._op_string[var._VarData]
                             % (self.ampl_var_id[self._varID_map[id(exp)]],
                                self._name_labeler(exp)))

        elif isinstance(exp,param._ParamData):
            OUTPUT.write(self._op_string[param._ParamData]
                         % (value(exp)))

        elif isinstance(exp,NumericConstant) or exp.is_fixed():
            OUTPUT.write(self._op_string[NumericConstant]
                         % (value(exp)))

        else:
            raise ValueError(
                "Unsupported expression type (%s) in _print_nonlinear_terms_NL"
                % (exp_type))

    def _print_model_NL(self, model,
                        solver_capability,
                        show_section_timing=False,
                        skip_trivial_constraints=False,
                        file_determinism=1,
                        include_all_variable_bounds=False):

        output_fixed_variable_bounds = self._output_fixed_variable_bounds
        symbolic_solver_labels = self._symbolic_solver_labels

        sorter = SortComponents.unsorted
        if file_determinism >= 1:
            sorter = sorter | SortComponents.indices
            if file_determinism >= 2:
                sorter = sorter | SortComponents.alphabetical

        OUTPUT = self._OUTPUT
        assert OUTPUT is not None

        # maps NL variables to the "real" variable names in the problem.
        # it's really NL variable ordering, as there are no variable names
        # in the NL format. however, we by convention make them go from
        # x0 upward.

        overall_timer = StopWatch()
        subsection_timer = StopWatch()

        # create the symbol_map
        symbol_map = SymbolMap()

        name_labeler = self._name_labeler
        # These will get updated when symbolic_solver_labels
        # is true. It is critical that nonzero values go in
        # the header of the NL file if you want ASL to
        # parse the .row and .col files.
        max_rowname_len = 0
        max_colname_len = 0

        #
        # Collect statistics
        #
        self_ampl_var_id = self.ampl_var_id = {}
        self_ampl_con_id = self.ampl_con_id = {}
        self_ampl_obj_id = self.ampl_obj_id = {}

        # will be the entire list of all vars in the model (not
        # necessarily used)
        Vars_dict = dict()
        # will be the entire list of all objective used in the problem
        # (size=1)
        Objectives_dict = dict()
        # will be the entire list of all constraints used in the
        # problem
        Constraints_dict = dict()

        UsedVars = set()

        # linear variables
        LinearVars = set()
        LinearVarsInt = set()
        LinearVarsBool = set()

        # Tabulate the External Function definitions
        self.external_byFcn = {}
        external_Libs = set()
        for fcn in model.component_objects(ExternalFunction, active=True):
            if fcn._function in self.external_byFcn:
                if self.external_byFcn[fcn._function][0]._library != fcn._library:
                    raise RuntimeError(
                        "The same external function name (%s) is associated "
                        "with two different libraries (%s through %s, and %s "
                        "through %s).  The ASL solver will fail to link "
                        "correctly." %
                        (fcn._function,
                         self.external_byFcn[fcn._function]._library,
                         self.external_byFcn[fcn._function]._library.name,
                         fcn._library,
                         fcn.name))
            else:
                self.external_byFcn[fcn._function] = \
                    (fcn, len(self.external_byFcn))
            external_Libs.add(fcn._library)
        if external_Libs:
            # The ASL AMPLFUNC environment variable is nominally a
            # whitespace-separated string of library names.  Beginning
            # sometime between 2010 and 2012, the ASL added support for
            # simple quoted strings: the first non-whitespace character
            # can be either " or '.  When that is detected, the ASL
            # parser will continue to the next occurance of that
            # character (i.e., no escaping is allowed).  We will use
            # that same logic here to quote any strings with spaces
            # ... bearing in mind that this will only work with solvers
            # compiled against versions of the ASL more recent than
            # ~2012.
            #
            # We are (arbitrarily) chosing to use newline as the field
            # separator.
            env_str = ''
            for _lib in external_Libs:
                _lib = _lib.strip()
                if ( ' ' not in _lib
                     or ( _lib[0]=='"' and _lib[-1]=='"'
                          and '"' not in _lib[1:-1] )
                     or ( _lib[0]=="'" and _lib[-1]=="'"
                          and "'" not in _lib[1:-1] ) ):
                    pass
                elif '"' not in _lib:
                    _lib = '"' + _lib + '"'
                elif "'" not in _lib:
                    _lib = "'" + _lib + "'"
                else:
                    raise RuntimeError(
                        "Cannot pass the AMPL external function library\n\t%s\n"
                        "to the ASL because the string contains spaces, "
                        "single quote and\ndouble quote characters." % (_lib,))
                if env_str:
                    env_str += "\n"
                env_str += _lib
            os.environ["PYOMO_AMPLFUNC"] = env_str
        elif "PYOMO_AMPLFUNC" in os.environ:
            del os.environ["PYOMO_AMPLFUNC"]

        subsection_timer.reset()

        # Cache the list of model blocks so we don't have to call
        # model.block_data_objects() many many times
        all_blocks_list = list(model.block_data_objects(active=True, sort=sorter))

        # create a deterministic var labeling
        Vars_dict = dict( enumerate( model.component_data_objects(
                    Var, sort=sorter) ) )
        cntr = len(Vars_dict)
        # cntr = 0
        # for block in all_blocks_list:
        #     vars_counter = tuple(enumerate(
        #         block.component_data_objects(Var,
        #                                      active=True,
        #                                      sort=sorter,
        #                                      descend_into=False),
        #         cntr))
        #     cntr += len(vars_counter)
        #     Vars_dict.update(vars_counter)
        self._varID_map = dict((id(val),key) for key,val in iteritems(Vars_dict))
        self_varID_map = self._varID_map
        # Use to label the rest of the components (which we will not encounter twice)
        trivial_labeler = _Counter(cntr)

        #
        # Count number of objectives and build the repns
        #
        n_objs = 0
        n_nonlinear_objs = 0
        ObjVars = set()
        ObjNonlinearVars = set()
        ObjNonlinearVarsInt = set()
        for block in all_blocks_list:

            gen_obj_repn = \
                getattr(block, "_gen_obj_repn", True)

            # Get/Create the ComponentMap for the repn
            if not hasattr(block,'_repn'):
                block._repn = ComponentMap()
            block_repn = block._repn

            for active_objective in block.component_data_objects(Objective,
                                                                 active=True,
                                                                 sort=sorter,
                                                                 descend_into=False):
                if symbolic_solver_labels:
                    objname = name_labeler(active_objective)
                    if len(objname) > max_rowname_len:
                        max_rowname_len = len(objname)

                if gen_obj_repn:
                    repn = generate_standard_repn(active_objective.expr,
                                                  quadratic=False)
                    block_repn[active_objective] = repn
                    linear_vars = repn.linear_vars
                    nonlinear_vars = repn.nonlinear_vars
                else:
                    repn = block_repn[active_objective]
                    linear_vars = repn.linear_vars
                    # By default, the NL writer generates
                    # StandardRepn objects without the more
                    # expense quadratic processing, but
                    # there is no guarantee of this if we
                    # are using a cached repn object, so we
                    # must check for the quadratic form.
                    if repn.is_nonlinear() and (repn.nonlinear_expr is None):
                        assert repn.is_quadratic()
                        assert len(repn.quadratic_vars) > 0
                        nonlinear_vars = {}
                        for v1, v2 in repn.quadratic_vars:
                            nonlinear_vars[id(v1)] = v1
                            nonlinear_vars[id(v2)] = v2
                        nonlinear_vars = nonlinear_vars.values()
                    else:
                        nonlinear_vars = repn.nonlinear_vars

                try:
                    wrapped_repn = RepnWrapper(
                        repn,
                        list(self_varID_map[id(var)] for var in linear_vars),
                        list(self_varID_map[id(var)] for var in nonlinear_vars))
                except KeyError as err:
                    self._symbolMapKeyError(err, model, self_varID_map,
                                            list(linear_vars) +
                                            list(nonlinear_vars))
                    raise

                LinearVars.update(wrapped_repn.linear_vars)
                ObjNonlinearVars.update(wrapped_repn.nonlinear_vars)

                ObjVars.update(wrapped_repn.linear_vars)
                ObjVars.update(wrapped_repn.nonlinear_vars)

                obj_ID = trivial_labeler(active_objective)
                Objectives_dict[obj_ID] = (active_objective, wrapped_repn)
                self_ampl_obj_id[obj_ID] = n_objs
                symbol_map.addSymbols([(active_objective, "o%d"%n_objs)])

                n_objs += 1
                if repn.is_nonlinear():
                    n_nonlinear_objs += 1

        # I don't think this is necessarily true for the entire code base,
        # but seeing has how its never been tested we should go ahead and
        # raise an exception
        if n_objs > 1:
            raise ValueError(
                "The NL writer has detected multiple active objective functions "
                "on model %s, but currently only handles a single objective."
                % (model.name))
        elif n_objs == 1:
            symbol_map.alias(symbol_map.bySymbol["o0"](),"__default_objective__")

        if show_section_timing:
            subsection_timer.report("Generate objective representation")
            subsection_timer.reset()

        #
        # Count number of constraints and build the repns
        #
        n_ranges = 0
        n_single_sided_ineq = 0
        n_equals = 0
        n_unbounded = 0
        n_nonlinear_constraints = 0
        ConNonlinearVars = set()
        ConNonlinearVarsInt = set()
        nnz_grad_constraints = 0
        constraint_bounds_dict = {}
        nonlin_con_order_list = []
        lin_con_order_list = []
        ccons_lin = 0
        ccons_nonlin = 0
        ccons_nd = 0
        ccons_nzlb = 0

        for block in all_blocks_list:
            all_repns = list()

            gen_con_repn = \
                getattr(block, "_gen_con_repn", True)

            # Get/Create the ComponentMap for the repn
            if not hasattr(block,'_repn'):
                block._repn = ComponentMap()
            block_repn = block._repn

            # Initializing the constraint dictionary
            for constraint_data in block.component_data_objects(Constraint,
                                                                active=True,
                                                                sort=sorter,
                                                                descend_into=False):

                if (not constraint_data.has_lb()) and \
                   (not constraint_data.has_ub()):
                    assert not constraint_data.equality
                    continue  # non-binding, so skip

                if symbolic_solver_labels:
                    conname = name_labeler(constraint_data)
                    if len(conname) > max_rowname_len:
                        max_rowname_len = len(conname)

                if constraint_data._linear_canonical_form:
                    repn = constraint_data.canonical_form()
                    linear_vars = repn.linear_vars
                    nonlinear_vars = repn.nonlinear_vars
                else:
                    if gen_con_repn:
                        repn = generate_standard_repn(constraint_data.body,
                                                      quadratic=False)
                        block_repn[constraint_data] = repn
                        linear_vars = repn.linear_vars
                        nonlinear_vars = repn.nonlinear_vars
                    else:
                        repn = block_repn[constraint_data]
                        linear_vars = repn.linear_vars
                        # By default, the NL writer generates
                        # StandardRepn objects without the more
                        # expense quadratic processing, but
                        # there is no guarantee of this if we
                        # are using a cached repn object, so we
                        # must check for the quadratic form.
                        if repn.is_nonlinear() and (repn.nonlinear_expr is None):
                            assert repn.is_quadratic()
                            assert len(repn.quadratic_vars) > 0
                            nonlinear_vars = {}
                            for v1, v2 in repn.quadratic_vars:
                                nonlinear_vars[id(v1)] = v1
                                nonlinear_vars[id(v2)] = v2
                            nonlinear_vars = nonlinear_vars.values()
                        else:
                            nonlinear_vars = repn.nonlinear_vars

                ### GAH: Even if this is fixed, it is still useful to
                ###      write out these types of constraints
                ###      (trivial) as a feasibility check for fixed
                ###      variables, in which case the solver will pick
                ###      up on the model infeasibility.
                if skip_trivial_constraints and repn.is_fixed():
                    continue

                con_ID = trivial_labeler(constraint_data)
                try:
                    wrapped_repn = RepnWrapper(
                        repn,
                        list(self_varID_map[id(var)] for var in linear_vars),
                        list(self_varID_map[id(var)] for var in nonlinear_vars))
                except KeyError as err:
                    self._symbolMapKeyError(err, model, self_varID_map,
                                            list(linear_vars) +
                                            list(nonlinear_vars))
                    raise

                if repn.is_nonlinear():
                    nonlin_con_order_list.append(con_ID)
                    n_nonlinear_constraints += 1
                else:
                    lin_con_order_list.append(con_ID)

                Constraints_dict[con_ID] = (constraint_data, wrapped_repn)

                LinearVars.update(wrapped_repn.linear_vars)
                ConNonlinearVars.update(wrapped_repn.nonlinear_vars)

                nnz_grad_constraints += \
                    len(set(wrapped_repn.linear_vars).union(
                        wrapped_repn.nonlinear_vars))

                L = None
                U = None
                if constraint_data.has_lb():
                    L = _get_bound(constraint_data.lower)
                else:
                    assert constraint_data.has_ub()
                if constraint_data.has_ub():
                    U = _get_bound(constraint_data.upper)
                else:
                    assert constraint_data.has_lb()
                if constraint_data.equality:
                    assert L == U

                offset = repn.constant
                _type = getattr(constraint_data, '_complementarity', None)
                _vid = getattr(constraint_data, '_vid', None)
                if not _type is None:
                    _vid = self_varID_map[_vid]+1
                    constraint_bounds_dict[con_ID] = \
                        "5 {0} {1}\n".format(_type, _vid)
                    if _type == 1 or _type == 2:
                        n_single_sided_ineq += 1
                    elif _type == 3:
                        n_ranges += 1
                    elif _type == 4:
                        n_unbounded += 1
                    if repn.is_nonlinear():
                        ccons_nonlin += 1
                    else:
                        ccons_lin += 1
                else:
                    if L == U:
                        if L is None:
                            # No constraint on body
                            constraint_bounds_dict[con_ID] = "3\n"
                            n_unbounded += 1
                        else:
                            constraint_bounds_dict[con_ID] = \
                                "4 %r\n" % (L-offset)
                            n_equals += 1
                    elif L is None:
                        constraint_bounds_dict[con_ID] = "1 %r\n" % (U-offset)
                        n_single_sided_ineq += 1
                    elif U is None:
                        constraint_bounds_dict[con_ID] = "2 %r\n" % (L-offset)
                        n_single_sided_ineq += 1
                    elif (L > U):
                        msg = 'Constraint {0}: lower bound greater than upper' \
                            ' bound ({1} > {2})'
                        raise ValueError(msg.format(constraint_data.name,
                                                    str(L), str(U)))
                    else:
                        constraint_bounds_dict[con_ID] = \
                            "0 %r %r\n" % (L-offset, U-offset)
                        # double sided inequality
                        # both are not none and they are valid
                        n_ranges += 1

        sos1 = solver_capability("sos1")
        sos2 = solver_capability("sos2")
        for block in all_blocks_list:
            for soscondata in block.component_data_objects(SOSConstraint,
                                                           active=True,
                                                           sort=sorter,
                                                           descend_into=False):
                level = soscondata.level
                if (level == 1 and not sos1) or (level == 2 and not sos2):
                    raise Exception(
                        "Solver does not support SOS level %s constraints"
                        % (level,))
                if hasattr(soscondata, "get_variables"):
                    LinearVars.update(self_varID_map[id(vardata)]
                                      for vardata in soscondata.get_variables())
                else:
                    LinearVars.update(self_varID_map[id(vardata)]
                                      for vardata in soscondata.variables)

        # create the ampl constraint ids
        self_ampl_con_id.update(
            (con_ID,row_id) for row_id,con_ID in \
            enumerate(itertools.chain(nonlin_con_order_list,lin_con_order_list)))
        # populate the symbol_map
        symbol_map.addSymbols(
            [(Constraints_dict[con_ID][0],"c%d"%row_id) for row_id,con_ID in \
             enumerate(itertools.chain(nonlin_con_order_list,lin_con_order_list))])

        if show_section_timing:
            subsection_timer.report("Generate constraint representations")
            subsection_timer.reset()

        UsedVars.update(LinearVars)
        UsedVars.update(ObjNonlinearVars)
        UsedVars.update(ConNonlinearVars)

        LinearVars = LinearVars.difference(ObjNonlinearVars)
        LinearVars = LinearVars.difference(ConNonlinearVars)

        if include_all_variable_bounds:
            # classify unused vars as linear
            AllVars = set(self_varID_map[id(vardata)]
                          for vardata in itervalues(Vars_dict))
            UnusedVars = AllVars.difference(UsedVars)
            LinearVars.update(UnusedVars)

        ### There used to be an if statement here for the following code block
        ### checking model.statistics.num_binary_vars was greater than zero.
        ### To this day, I don't know how it worked.
        ### TODO: Figure out why
        ###############

        for var_ID in LinearVars:
            var = Vars_dict[var_ID]
            if var.is_binary():
                L = var.lb
                U = var.ub
                if (L is None) or (U is None):
                    raise ValueError("Variable " + str(var.name) +\
                                     "is binary, but does not have lb and ub set")
                LinearVarsBool.add(var_ID)
            elif var.is_integer():
                LinearVarsInt.add(var_ID)
            elif not var.is_continuous():
                raise TypeError("Invalid domain type for variable with name '%s'. "
                                "Variable is not continuous, integer, or binary.")
        LinearVars.difference_update(LinearVarsInt)
        LinearVars.difference_update(LinearVarsBool)

        for var_ID in ObjNonlinearVars:
            var = Vars_dict[var_ID]
            if var.is_integer() or var.is_binary():
                ObjNonlinearVarsInt.add(var_ID)
            elif not var.is_continuous():
                raise TypeError("Invalid domain type for variable with name '%s'. "
                                "Variable is not continuous, integer, or binary.")
        ObjNonlinearVars.difference_update(ObjNonlinearVarsInt)
        for var_ID in ConNonlinearVars:
            var = Vars_dict[var_ID]
            if var.is_integer() or var.is_binary():
                ConNonlinearVarsInt.add(var_ID)
            elif not var.is_continuous():
                raise TypeError("Invalid domain type for variable with name '%s'. "
                                "Variable is not continuous, integer, or binary.")
        ConNonlinearVars.difference_update(ConNonlinearVarsInt)
        ##################

        Nonlinear_Vars_in_Objs_and_Constraints = \
            ObjNonlinearVars.intersection(ConNonlinearVars)
        Discrete_Nonlinear_Vars_in_Objs_and_Constraints = \
            ObjNonlinearVarsInt.intersection(ConNonlinearVarsInt)
        ObjNonlinearVars = \
            ObjNonlinearVars.difference(Nonlinear_Vars_in_Objs_and_Constraints)
        ConNonlinearVars = \
            ConNonlinearVars.difference(Nonlinear_Vars_in_Objs_and_Constraints)
        ObjNonlinearVarsInt = \
            ObjNonlinearVarsInt.difference(
                Discrete_Nonlinear_Vars_in_Objs_and_Constraints)
        ConNonlinearVarsInt = \
            ConNonlinearVarsInt.difference(
                Discrete_Nonlinear_Vars_in_Objs_and_Constraints)

        # put the ampl variable id into the variable
        full_var_list = []
        full_var_list.extend(sorted(Nonlinear_Vars_in_Objs_and_Constraints))
        full_var_list.extend(sorted(Discrete_Nonlinear_Vars_in_Objs_and_Constraints))
        idx_nl_both = len(full_var_list)
        #
        full_var_list.extend(sorted(ConNonlinearVars))
        full_var_list.extend(sorted(ConNonlinearVarsInt))
        idx_nl_con = len(full_var_list)
        #
        full_var_list.extend(sorted(ObjNonlinearVars))
        full_var_list.extend(sorted(ObjNonlinearVarsInt))
        idx_nl_obj = len(full_var_list)
        #
        full_var_list.extend(sorted(LinearVars))
        full_var_list.extend(sorted(LinearVarsBool))
        full_var_list.extend(sorted(LinearVarsInt))

        if (idx_nl_obj == idx_nl_con):
            idx_nl_obj = idx_nl_both

        # create the ampl variable column ids
        self_ampl_var_id.update((var_ID,column_id)
                                for column_id,var_ID in enumerate(full_var_list))
        # populate the symbol_map
        symbol_map.addSymbols([(Vars_dict[var_ID],"v%d"%column_id)
                               for column_id,var_ID in enumerate(full_var_list)])

        if show_section_timing:
            subsection_timer.report("Partition variable types")
            subsection_timer.reset()

#        end_time = time.clock()
#        print (end_time - start_time)

        colfilename = None
        if OUTPUT.name.endswith('.nl'):
            colfilename = OUTPUT.name.replace('.nl','.col')
        else:
            colfilename = OUTPUT.name+'.col'
        if symbolic_solver_labels:
            colf = open(colfilename,'w')
            colfile_line_template = "%s\n"
            for var_ID in full_var_list:
                varname = name_labeler(Vars_dict[var_ID])
                colf.write(colfile_line_template % varname)
                if len(varname) > max_colname_len:
                    max_colname_len = len(varname)
            colf.close()

        if show_section_timing:
            subsection_timer.report("Write .col file")
            subsection_timer.reset()

        #
        # Print Header
        #
        # LINE 1
        #
        OUTPUT.write("g3 1 1 0\t# problem {0}\n".format(model.name))
        #
        # LINE 2
        #
        OUTPUT.write(" {0} {1} {2} {3} {4} \t# vars, constraints, "
                     "objectives, ranges, eqns\n" .format(
                         len(full_var_list),
                         n_single_sided_ineq + n_ranges+n_equals+n_unbounded,
                         n_objs,
                         n_ranges,
                         n_equals))
        #
        # LINE 3
        #
        OUTPUT.write(" {0} {1} {2} {3} {4} {5}\t# nonlinear constrs, "
                     "objs; ccons: lin, nonlin, nd, nzlb\n".format(
                         n_nonlinear_constraints,
                         n_nonlinear_objs,
                         ccons_lin,
                         ccons_nonlin,
                         ccons_nd,
                         ccons_nzlb))
        #
        # LINE 4
        #
        OUTPUT.write(" 0 0\t# network constraints: nonlinear, linear\n")
        #
        # LINE 5
        #
        OUTPUT.write(" {0} {1} {2} \t# nonlinear vars in constraints, "
                     "objectives, both\n".format(
                         idx_nl_con,
                         idx_nl_obj,
                         idx_nl_both))

        #
        # LINE 6
        #
        OUTPUT.write(" 0 {0} 0 1\t# linear network variables; functions; "
                     "arith, flags\n".format(len(self.external_byFcn)))
        #
        # LINE 7
        #
        n_int_nonlinear_b = len(Discrete_Nonlinear_Vars_in_Objs_and_Constraints)
        n_int_nonlinear_c = len(ConNonlinearVarsInt)
        n_int_nonlinear_o = len(ObjNonlinearVarsInt)
        OUTPUT.write(" {0} {1} {2} {3} {4} \t# discrete variables: binary, "
                     "integer, nonlinear (b,c,o)\n".format(
                         len(LinearVarsBool),
                         len(LinearVarsInt),
                         n_int_nonlinear_b,
                         n_int_nonlinear_c,
                         n_int_nonlinear_o))
        #
        # LINE 8
        #
        # objective info computed above
        OUTPUT.write(" {0} {1} \t# nonzeros in Jacobian, obj. gradient\n".format(
            nnz_grad_constraints,
            len(ObjVars)))
        #
        # LINE 9
        #
        OUTPUT.write(" %d %d\t# max name lengths: constraints, variables\n"
                     % (max_rowname_len, max_colname_len))

        #
        # LINE 10
        #
        OUTPUT.write(" 0 0 0 0 0\t# common exprs: b,c,o,c1,o1\n")

#        end_time = time.clock()
#        print (end_time - start_time)

#        print "Printing constraints:",
#        start_time = time.clock()


        #
        # "F" lines
        #
        for fcn, fid in sorted(itervalues(self.external_byFcn),
                               key=operator.itemgetter(1)):
            OUTPUT.write("F%d 1 -1 %s\n" % (fid, fcn._function))

        #
        # "S" lines
        #

        # Tranlate the SOSConstraint component into ampl suffixes
        sos1 = solver_capability("sos1")
        sos2 = solver_capability("sos2")
        modelSOS = ModelSOS(self_ampl_var_id, self_varID_map)
        for block in all_blocks_list:
            for soscondata in block.component_data_objects(SOSConstraint,
                                                           active=True,
                                                           sort=sorter,
                                                           descend_into=False):
                level = soscondata.level
                if (level == 1 and not sos1) or (level == 2 and not sos2):
                    raise ValueError(
                        "Solver does not support SOS level %s constraints" % (level))
                modelSOS.count_constraint(soscondata)

        symbol_map_byObject = symbol_map.byObject

        var_sosno_suffix = modelSOS.sosno
        var_ref_suffix = modelSOS.ref
        sosconstraint_sosno_vals = set(var_sosno_suffix.vals)

        # Translate the rest of the Pyomo Suffix components
        suffix_header_line = "S{0} {1} {2}\n"
        suffix_line = "{0} {1!r}\n"
        var_tag = 0
        con_tag = 1
        obj_tag = 2
        prob_tag = 3
        suffix_dict = {}
        if isinstance(model, IBlock):
            suffix_gen = lambda b: ((suf.storage_key, suf) \
                                    for suf in pyomo.core.kernel.suffix.\
                                    export_suffix_generator(b,
                                                            active=True,
                                                            descend_into=False))
        else:
            suffix_gen = lambda b: pyomo.core.base.suffix.\
                         active_export_suffix_generator(b)
        for block in all_blocks_list:
            for name, suf in suffix_gen(block):
                if len(suf):
                    suffix_dict.setdefault(name,[]).append(suf)
        if not ('sosno' in suffix_dict):
            # We still need to write out the SOSConstraint suffixes
            # even though these may have not been "declared" on the model
            s_lines = var_sosno_suffix.genfilelines()
            len_s_lines = len(s_lines)
            if len_s_lines > 0:
                OUTPUT.write(suffix_header_line.format(var_tag,len_s_lines,'sosno'))
                OUTPUT.writelines(s_lines)
        else:
            # I am choosing not to allow a user to mix the use of the Pyomo
            # SOSConstraint component and manual sosno declarations within
            # a single model. I initially tried to allow this but the
            # var section of the code below blows up for two reason. (1)
            # we have to make sure that the sosno suffix is not defined
            # twice for the same variable (2) We have to make sure that
            # the automatically chosen sosno used by the SOSConstraint
            # translation does not already match one a user has manually
            # implemented (this would modify the members in an sos set).
            # Since this suffix is exclusively used for defining sos sets,
            # there is no reason a user can not just stick to one method.
            if not var_sosno_suffix.is_empty():
                raise RuntimeError(
                    "The Pyomo NL file writer does not allow both manually "
                    "declared 'sosno' suffixes as well as SOSConstraint "
                    "components to exist on a single model. To avoid this "
                    "error please use only one of these methods to define "
                    "special ordered sets.")
        if not ('ref' in suffix_dict):
            # We still need to write out the SOSConstraint suffixes
            # even though these may have not been "declared" on the model
            s_lines = var_ref_suffix.genfilelines()
            len_s_lines = len(s_lines)
            if len_s_lines > 0:
                OUTPUT.write(suffix_header_line.format(var_tag,len_s_lines,'ref'))
                OUTPUT.writelines(s_lines)
        else:
            # see reason (1) in the paragraph above for why we raise this
            # exception (replacing sosno with ref).
            if not var_ref_suffix.is_empty():
                raise RuntimeError(
                    "The Pyomo NL file writer does not allow both manually "
                    "declared 'ref' suffixes as well as SOSConstraint "
                    "components to exist on a single model. To avoid this "
                    "error please use only one of these methods to define "
                    "special ordered sets.")
        # do a sort to make sure NL file output is deterministic
        # across python versions
        for suffix_name in sorted(suffix_dict):
            suffixes = suffix_dict[suffix_name]
            datatypes = set()
            for suffix in suffixes:
                try:
                    datatype = suffix.datatype
                except AttributeError:
                    datatype = suffix.get_datatype()
                if datatype not in (Suffix.FLOAT,Suffix.INT):
                    raise ValueError(
                        "The Pyomo NL file writer requires that all active export "
                        "Suffix components declare a numeric datatype. Suffix "
                        "component: %s with " % (suffix_name))
                datatypes.add(datatype)
            if len(datatypes) != 1:
                raise ValueError(
                    "The Pyomo NL file writer found multiple active export suffix "
                    "components with name %s with different datatypes. A single "
                    "datatype must be declared." % (suffix_name))
            if suffix_name == "dual":
                # The NL file format has a special section for dual initializations
                continue
            float_tag = 0
            if datatypes.pop() == Suffix.FLOAT:
                float_tag = 4

            var_s_lines = []
            con_s_lines = []
            obj_s_lines = []
            mod_s_lines = []
            for suffix in suffixes:
                for component_data, suffix_value in iteritems(suffix):

                    try:
                        symbol = symbol_map_byObject[id(component_data)]
                        type_tag = symbol[0]
                        ampl_id = int(symbol[1:])
                        if type_tag == 'v':
                            var_s_lines.append((ampl_id, suffix_value))
                        elif type_tag == 'c':
                            con_s_lines.append((ampl_id, suffix_value))
                        elif type_tag == 'o':
                            obj_s_lines.append((ampl_id, suffix_value))
                        else:
                            # This would be a developer error
                            assert False
                    except KeyError:
                        if component_data is model:
                            mod_s_lines.append((0, suffix_value))

            ################## vars
            if len(var_s_lines) > 0:
                OUTPUT.write(suffix_header_line.format(var_tag | float_tag,
                                                       len(var_s_lines),
                                                       suffix_name))
                OUTPUT.writelines(suffix_line.format(*_l)
                                  for _l in sorted(var_s_lines,
                                                   key=operator.itemgetter(0)))
            ################## constraints
            if len(con_s_lines) > 0:
                OUTPUT.write(suffix_header_line.format(con_tag | float_tag,
                                                       len(con_s_lines),
                                                       suffix_name))
                OUTPUT.writelines(suffix_line.format(*_l)
                                  for _l in sorted(con_s_lines,
                                                   key=operator.itemgetter(0)))
            ################## objectives
            if len(obj_s_lines) > 0:
                OUTPUT.write(suffix_header_line.format(obj_tag | float_tag,
                                                       len(obj_s_lines),
                                                       suffix_name))
                OUTPUT.writelines(suffix_line.format(*_l)
                                  for _l in sorted(obj_s_lines,
                                                   key=operator.itemgetter(0)))
            ################## problems (in this case the one problem)
            if len(mod_s_lines) > 0:
                if len(mod_s_lines) > 1:
                    logger.warning(
                        "ProblemWriter_nl: Collected multiple values for Suffix %s "
                        "referencing model %s. This is likely a bug."
                        % (suffix_name, model.name))
                OUTPUT.write(suffix_header_line.format(prob_tag | float_tag,
                                                       len(mod_s_lines),
                                                       suffix_name))
                OUTPUT.writelines(suffix_line.format(*_l)
                                  for _l in sorted(mod_s_lines,
                                                   key=operator.itemgetter(0)))

        del modelSOS

        #
        # "C" lines
        #
        rowfilename = None
        if OUTPUT.name.endswith('.nl'):
            rowfilename = OUTPUT.name.replace('.nl','.row')
        else:
            rowfilename = OUTPUT.name+'.row'
        if symbolic_solver_labels:
            rowf = open(rowfilename,'w')

        cu = [0 for i in xrange(len(full_var_list))]
        for con_ID in nonlin_con_order_list:
            con_data, wrapped_repn = Constraints_dict[con_ID]
            row_id = self_ampl_con_id[con_ID]
            OUTPUT.write("C%d" % (row_id))
            if symbolic_solver_labels:
                lbl = name_labeler(con_data)
                OUTPUT.write("\t#%s" % (lbl))
                rowf.write(lbl+"\n")
            OUTPUT.write("\n")

            if wrapped_repn.repn.nonlinear_expr is not None:
                assert not wrapped_repn.repn.is_quadratic()
                self._print_nonlinear_terms_NL(
                    wrapped_repn.repn.nonlinear_expr)
            else:
                assert wrapped_repn.repn.is_quadratic()
                self._print_standard_quadratic_NL(
                    wrapped_repn.repn.quadratic_vars,
                    wrapped_repn.repn.quadratic_coefs)

            for var_ID in set(wrapped_repn.linear_vars).union(
                    wrapped_repn.nonlinear_vars):
                cu[self_ampl_var_id[var_ID]] += 1

        for con_ID in lin_con_order_list:
            con_data, wrapped_repn = Constraints_dict[con_ID]
            row_id = self_ampl_con_id[con_ID]
            con_vars = set(wrapped_repn.linear_vars)
            for var_ID in con_vars:
                cu[self_ampl_var_id[var_ID]] += 1
            OUTPUT.write("C%d" % (row_id))
            if symbolic_solver_labels:
                lbl = name_labeler(con_data)
                OUTPUT.write("\t#%s" % (lbl))
                rowf.write(lbl+"\n")
            OUTPUT.write("\n")
            OUTPUT.write("n0\n")

        if show_section_timing:
            subsection_timer.report("Write NL header and suffix lines")
            subsection_timer.reset()

        #
        # "O" lines
        #
        for obj_ID, (obj, wrapped_repn) in iteritems(Objectives_dict):

            k = 0
            if not obj.is_minimizing():
                k = 1

            OUTPUT.write("O%d %d" % (self_ampl_obj_id[obj_ID], k))
            if symbolic_solver_labels:
                lbl = name_labeler(obj)
                OUTPUT.write("\t#%s" % (lbl))
                rowf.write(lbl+"\n")
            OUTPUT.write("\n")

            if wrapped_repn.repn.is_linear():
                OUTPUT.write(self._op_string[NumericConstant]
                             % (wrapped_repn.repn.constant))
            else:
                if wrapped_repn.repn.constant != 0:
                    _, binary_sum_str, _ = self._op_string[EXPR.SumExpressionBase]
                    OUTPUT.write(binary_sum_str)
                    OUTPUT.write(self._op_string[NumericConstant]
                                 % (wrapped_repn.repn.constant))
                if wrapped_repn.repn.nonlinear_expr is not None:
                    assert not wrapped_repn.repn.is_quadratic()
                    self._print_nonlinear_terms_NL(
                        wrapped_repn.repn.nonlinear_expr)
                else:
                    assert wrapped_repn.repn.is_quadratic()
                    self._print_standard_quadratic_NL(
                        wrapped_repn.repn.quadratic_vars,
                        wrapped_repn.repn.quadratic_coefs)

        if symbolic_solver_labels:
            rowf.close()
        del name_labeler

        if show_section_timing:
            subsection_timer.report("Write objective expression")
            subsection_timer.reset()

        #
        # "d" lines
        #
        # dual initialization
        if 'dual' in suffix_dict:
            s_lines = []
            for dual_suffix in suffix_dict['dual']:

                for constraint_data, suffix_value in iteritems(dual_suffix):
                    try:
                        # a constraint might not be referenced
                        # (inactive / on inactive block)
                        symbol = symbol_map_byObject[id(constraint_data)]
                        type_tag = symbol[0]
                        assert type_tag == 'c'
                        ampl_con_id = int(symbol[1:])
                        s_lines.append((ampl_con_id, suffix_value))
                    except KeyError:
                        pass

            if len(s_lines) > 0:
                OUTPUT.write("d%d" % (len(s_lines)))
                if symbolic_solver_labels:
                    OUTPUT.write("\t# dual initial guess")
                OUTPUT.write("\n")
                OUTPUT.writelines(suffix_line.format(*_l)
                                  for _l in sorted(s_lines,
                                                   key=operator.itemgetter(0)))

        #
        # "x" lines
        #
        # variable initialization
        var_bound_list = []
        x_init_list = []
        for ampl_var_id, var_ID in enumerate(full_var_list):
            var = Vars_dict[var_ID]
            if var.value is not None:
                x_init_list.append("%d %r\n" % (ampl_var_id, var.value))
            if var.fixed:
                if not output_fixed_variable_bounds:
                    raise ValueError(
                        "Encountered a fixed variable (%s) inside an active objective"
                        " or constraint expression on model %s, which is usually "
                        "indicative of a preprocessing error. Use the IO-option "
                        "'output_fixed_variable_bounds=True' to suppress this error "
                        "and fix the variable by overwriting its bounds in the NL "
                        "file." % (var.name, model.name))
                if var.value is None:
                    raise ValueError("Variable cannot be fixed to a value of None.")
                L = U = _get_bound(var.value)
            else:
                L = None
                if var.has_lb():
                    L = _get_bound(var.lb)
                U = None
                if var.has_ub():
                    U = _get_bound(var.ub)
            if L is not None:
                if U is not None:
                    if L == U:
                        var_bound_list.append("4 %r\n" % (L))
                    else:
                        var_bound_list.append("0 %r %r\n" % (L, U))
                else:
                    var_bound_list.append("2 %r\n" % (L))
            elif U is not None:
                var_bound_list.append("1 %r\n" % (U))
            else:
                var_bound_list.append("3\n")

        OUTPUT.write("x%d" % (len(x_init_list)))
        if symbolic_solver_labels:
            OUTPUT.write("\t# initial guess")
        OUTPUT.write("\n")
        OUTPUT.writelines(x_init_list)
        del x_init_list

        if show_section_timing:
            subsection_timer.report("Write initializations")
            subsection_timer.reset()

        #
        # "r" lines
        #
        OUTPUT.write("r")
        if symbolic_solver_labels:
            OUTPUT.write("\t#%d ranges (rhs's)"
                         % (len(nonlin_con_order_list) + len(lin_con_order_list)))
        OUTPUT.write("\n")
        # *NOTE: This iteration follows the assignment of the ampl_con_id
        OUTPUT.writelines(constraint_bounds_dict[con_ID]
                          for con_ID in itertools.chain(nonlin_con_order_list,
                                                        lin_con_order_list))

        if show_section_timing:
            subsection_timer.report("Write constraint bounds")
            subsection_timer.reset()

        #
        # "b" lines
        #
        OUTPUT.write("b")
        if symbolic_solver_labels:
            OUTPUT.write("\t#%d bounds (on variables)"
                         % (len(var_bound_list)))
        OUTPUT.write("\n")
        OUTPUT.writelines(var_bound_list)
        del var_bound_list

        if show_section_timing:
            subsection_timer.report("Write variable bounds")
            subsection_timer.reset()

        #
        # "k" lines
        #
        ktot = 0
        n1 = len(full_var_list) - 1
        OUTPUT.write("k%d" % (n1))
        if symbolic_solver_labels:
            OUTPUT.write("\t#intermediate Jacobian column lengths")
        OUTPUT.write("\n")
        ktot = 0
        for i in xrange(n1):
            ktot += cu[i]
            OUTPUT.write("%d\n"%(ktot))
        del cu

        if show_section_timing:
            subsection_timer.report("Write k lines")
            subsection_timer.reset()

        #
        # "J" lines
        #
        for nc, con_ID in enumerate(itertools.chain(nonlin_con_order_list,
                                                    lin_con_order_list)):
            con_data, wrapped_repn = Constraints_dict[con_ID]
            numnonlinear_vars = len(wrapped_repn.nonlinear_vars)
            numlinear_vars = len(wrapped_repn.linear_vars)
            if numnonlinear_vars == 0:
                if numlinear_vars > 0:
                    linear_dict = dict((var_ID, coef)
                                       for var_ID, coef in
                                       zip(wrapped_repn.linear_vars,
                                           wrapped_repn.repn.linear_coefs))
                    OUTPUT.write("J%d %d\n"%(nc, numlinear_vars))
                    OUTPUT.writelines(
                        "%d %r\n" % (self_ampl_var_id[con_var],
                                     linear_dict[con_var])
                        for con_var in sorted(linear_dict.keys()))
            elif numlinear_vars == 0:
                nl_con_vars = \
                    sorted(wrapped_repn.nonlinear_vars)
                OUTPUT.write("J%d %d\n"%(nc, numnonlinear_vars))
                OUTPUT.writelines(
                    "%d 0\n"%(self_ampl_var_id[con_var])
                    for con_var in nl_con_vars)
            else:
                con_vars = set(wrapped_repn.nonlinear_vars)
                nl_con_vars = sorted(
                    con_vars.difference(
                        wrapped_repn.linear_vars))
                con_vars.update(wrapped_repn.linear_vars)
                linear_dict = dict(
                    (var_ID, coef) for var_ID, coef in
                    zip(wrapped_repn.linear_vars,
                        wrapped_repn.repn.linear_coefs))
                OUTPUT.write("J%d %d\n"%(nc, len(con_vars)))
                OUTPUT.writelines(
                    "%d %r\n" % (self_ampl_var_id[con_var],
                                 linear_dict[con_var])
                    for con_var in sorted(linear_dict.keys()))
                OUTPUT.writelines(
                    "%d 0\n"%(self_ampl_var_id[con_var])
                    for con_var in nl_con_vars)


        if show_section_timing:
            subsection_timer.report("Write J lines")
            subsection_timer.reset()

        #
        # "G" lines
        #
        for obj_ID, (obj, wrapped_repn) in \
               iteritems(Objectives_dict):

            grad_entries = {}
            for idx, obj_var in enumerate(
                    wrapped_repn.linear_vars):
                grad_entries[self_ampl_var_id[obj_var]] = \
                    wrapped_repn.repn.linear_coefs[idx]
            for obj_var in wrapped_repn.nonlinear_vars:
                if obj_var not in wrapped_repn.linear_vars:
                    grad_entries[self_ampl_var_id[obj_var]] = 0
            len_ge = len(grad_entries)
            if len_ge > 0:
                OUTPUT.write("G%d %d\n" % (self_ampl_obj_id[obj_ID],
                                           len_ge))
                for var_ID in sorted(grad_entries.keys()):
                    OUTPUT.write("%d %r\n" % (var_ID,
                                              grad_entries[var_ID]))

        if show_section_timing:
            subsection_timer.report("Write G lines")
            subsection_timer.reset()
            overall_timer.report("Total time")

        return symbol_map

    def _symbolMapKeyError(self, err, model, map, vars):
        _errors = []
        for v in vars:
            if id(v) in map:
                continue
            if v.model() is not model.model():
                _errors.append(
                    "Variable '%s' is not part of the model "
                    "being written out, but appears in an "
                    "expression used on this model." % (v.name,))
            else:
                _parent = v.parent_block()
                while _parent is not None and _parent is not model:
                    if _parent.type() is not model.type():
                        _errors.append(
                            "Variable '%s' exists within %s '%s', "
                            "but is used by an active "
                            "expression.  Currently variables "
                            "must be reachable through a tree "
                            "of active Blocks."
                            % (v.name, _parent.type().__name__,
                               _parent.name))
                    if not _parent.active:
                        _errors.append(
                            "Variable '%s' exists within "
                            "deactivated %s '%s', but is used by "
                            "an active expression.  Currently "
                            "variables must be reachable through "
                            "a tree of active Blocks."
                            % (v.name, _parent.type().__name__,
                               _parent.name))
                    _parent = _parent.parent_block()

        if _errors:
            for e in _errors:
                logger.error(e)
            err.args = err.args + tuple(_errors)
