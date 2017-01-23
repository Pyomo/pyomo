#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

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

from pyutilib.misc import PauseGC
from pyutilib.math import infinity

import pyomo.util.plugin
from pyomo.opt import ProblemFormat
from pyomo.opt.base import *
from pyomo.core.base import *
from pyomo.core.base import expr, SymbolMap, Block
import pyomo.core.base.expr_common
from pyomo.core.base.var import Var
from pyomo.core.base import _ExpressionData, Expression, SortComponents
from pyomo.core.base.numvalue import NumericConstant, native_numeric_types
from pyomo.core.base import var
from pyomo.core.base import param
from pyomo.core.base.suffix import active_export_suffix_generator
from pyomo.repn.ampl_repn import generate_ampl_repn

from six import itervalues, iteritems
from six.moves import xrange, zip

_using_pyomo4_trees = False
if pyomo.core.base.expr_common.mode == \
   pyomo.core.base.expr_common.Mode.pyomo4_trees:
    _using_pyomo4_trees = True

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
    'abs':    'o15'}

# build string templates

_op_template = {}
_op_comment = {}

prod_template = "o2{C}\n"
prod_comment = "\t#*"
div_template = "o3{C}\n"
div_comment = "\t#/"
if _using_pyomo4_trees:
    _op_template[expr._ProductExpression] = prod_template
    _op_comment[expr._ProductExpression] = prod_comment
    _op_template[expr._DivisionExpression] = div_template
    _op_comment[expr._DivisionExpression] = div_comment
else:
    _op_template[expr._ProductExpression] = (prod_template,
                                             div_template)
    _op_comment[expr._ProductExpression] = (prod_comment,
                                            div_comment)
del prod_template
del prod_comment
del div_template
del div_comment

_op_template[expr._ExternalFunctionExpression] = ("f%d %d{C}\n", #function
                                                  "h%d:%s{C}\n") #string arg
_op_comment[expr._ExternalFunctionExpression] = ("\t#%s", #function
                                                 "")      #string arg

for opname in _intrinsic_function_operators:
    _op_template[opname] = _intrinsic_function_operators[opname]+"{C}\n"
    _op_comment[opname] = "\t#"+opname

_op_template[expr.Expr_if] = "o35{C}\n"
_op_comment[expr.Expr_if] = "\t#if"

_op_template[expr._InequalityExpression] = ("o21{C}\n", # and
                                            "o22{C}\n", # <
                                            "o23{C}\n") # <=
_op_comment[expr._InequalityExpression] = ("\t#and", # and
                                           "\t#lt",  # <
                                           "\t#le")  # <=

_op_template[expr._EqualityExpression] = "o24{C}\n"
_op_comment[expr._EqualityExpression] = "\t#eq"

_op_template[var._VarData] = "v%d{C}\n"
_op_comment[var._VarData] = "\t#%s"

_op_template[param._ParamData] = "n%r{C}\n"
_op_comment[param._ParamData] = ""

_op_template[NumericConstant] = "n%r{C}\n"
_op_comment[NumericConstant] = ""

_op_template[expr._SumExpression] = ("o54{C}\n%d\n", # nary +
                                     "o0{C}\n",      # +
                                     "o2\n" + _op_template[NumericConstant]) # * coef
_op_comment[expr._SumExpression] = ("\t#sumlist", # nary +
                                    "\t#+",       # +
                                    _op_comment[NumericConstant]) # * coef
if _using_pyomo4_trees:
    _op_template[expr._LinearExpression] = _op_template[expr._SumExpression]
    _op_comment[expr._LinearExpression] = _op_comment[expr._SumExpression]

    _op_template[expr._NegationExpression] = "o16{C}\n"
    _op_comment[expr._NegationExpression] = "\t#-"

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

        if soscondata.num_variables() == 0:
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

        for vardata, weight in soscondata.get_items():
            if vardata.fixed:
                raise RuntimeError(
                    "SOSConstraint '%s' includes a fixed Variable '%s'. "
                    "This is currently not supported. Deactivate this constraint "
                    "in order to proceed"
                    % (soscondata.name, vardata.name))

            ID = ampl_var_id[varID_map[id(vardata)]]
            self.sosno.add(ID,self.block_cntr*sign_tag)
            self.ref.add(ID,weight)

class RepnWrapper(object):

    __slots__ = ('repn','_linear_vars','_nonlinear_vars')

    def __init__(self,repn,linear,nonlinear):
        self.repn = repn
        self._linear_vars = linear
        self._nonlinear_vars = nonlinear

class ProblemWriter_nl(AbstractProblemWriter):

    pyomo.util.plugin.alias(str(ProblemFormat.nl),
                            'Generate the corresponding AMPL NL file.')

    def __init__(self):
        self._ampl_var_id = {}
        self._ampl_con_id = {}
        self._ampl_obj_id = {}
        self._OUTPUT = None
        self._varID_map = None
        AbstractProblemWriter.__init__(self, ProblemFormat.nl)

    def __call__(self,
                 model,
                 filename,
                 solver_capability,
                 io_options):

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

    def _get_bound(self, exp):
        if exp is None:
            return None
        if is_fixed(exp):
            return value(exp)
        raise ValueError("non-fixed bound: " + str(exp))

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
                self._op_string[expr._SumExpression]
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

        elif exp.is_expression():
            if _using_pyomo4_trees and (exp_type is expr._LinearExpression):
                nary_sum_str, binary_sum_str, coef_term_str = \
                    self._op_string[expr._LinearExpression]
                n = len(exp._args)
                if n > 2:
                    OUTPUT.write(nary_sum_str % (n))
                    if exp._const != 0:
                        OUTPUT.write(binary_sum_str)
                        OUTPUT.write(self._op_string[NumericConstant]
                                     % (exp._const))
                    for child_exp in exp._args:
                        child_coef = exp._coef[id(child_exp)]
                        if child_coef != 1:
                            OUTPUT.write(coef_term_str % (value(child_coef)))
                        self._print_nonlinear_terms_NL(child_exp)
                else:
                    assert n > 0
                    if exp._const != 0:
                        OUTPUT.write(binary_sum_str)
                        OUTPUT.write(self._op_string[NumericConstant]
                                     % (exp._const))
                    for i in xrange(n-1):
                        OUTPUT.write(binary_sum_str)
                        child_exp = exp._args[i]
                        child_coef = exp._coef[id(child_exp)]
                        if child_coef != 1:
                            OUTPUT.write(coef_term_str % (value(child_coef)))
                        self._print_nonlinear_terms_NL(child_exp)
                    child_exp = exp._args[n-1]
                    child_coef = exp._coef[id(child_exp)]
                    if child_coef != 1:
                        OUTPUT.write(coef_term_str % (value(child_coef)))
                    self._print_nonlinear_terms_NL(child_exp)

            elif _using_pyomo4_trees and (exp_type is expr._SumExpression):
                nary_sum_str, binary_sum_str, coef_term_str = \
                    self._op_string[expr._SumExpression]
                n = len(exp._args)
                if n > 2:
                    OUTPUT.write(nary_sum_str % (n))
                    for child_exp in exp._args:
                        self._print_nonlinear_terms_NL(child_exp)
                else:
                    for i in xrange(0,n-1):
                        OUTPUT.write(binary_sum_str)
                        self._print_nonlinear_terms_NL(exp._args[i])
                    self._print_nonlinear_terms_NL(exp._args[n-1])

            elif exp_type is expr._SumExpression:
                assert not _using_pyomo4_trees
                nary_sum_str, binary_sum_str, coef_term_str = \
                    self._op_string[expr._SumExpression]
                n = len(exp._args)
                if n > 2:
                    OUTPUT.write(nary_sum_str % (n))
                    if exp._const != 0:
                        OUTPUT.write(binary_sum_str)
                        OUTPUT.write(self._op_string[NumericConstant]
                                     % (exp._const))
                    for i in xrange(0,n):
                        if exp._coef[i] != 1:
                            OUTPUT.write(coef_term_str % (exp._coef[i]))
                        self._print_nonlinear_terms_NL(exp._args[i])
                else:
                    if exp._const != 0:
                        OUTPUT.write(binary_sum_str)
                        OUTPUT.write(self._op_string[NumericConstant]
                                     % (exp._const))
                    for i in xrange(0,n-1):
                        OUTPUT.write(binary_sum_str)
                        if exp._coef[i] != 1:
                            OUTPUT.write(coef_term_str % (exp._coef[i]))
                        self._print_nonlinear_terms_NL(exp._args[i])
                    if exp._coef[n-1] != 1:
                        OUTPUT.write(coef_term_str % (exp._coef[n-1]))
                    self._print_nonlinear_terms_NL(exp._args[n-1])

            elif (not _using_pyomo4_trees) and \
                 (exp_type is expr._ProductExpression):
                denom_exists = False
                prod_str, div_str = self._op_string[expr._ProductExpression]
                if len(exp._denominator) == 0:
                    pass
                else:
                    OUTPUT.write(div_str)
                    denom_exists = True
                if exp._coef != 1:
                    OUTPUT.write(prod_str)
                    OUTPUT.write(self._op_string[NumericConstant]
                                 % (exp._coef))
                if len(exp._numerator) == 0:
                    OUTPUT.write("n1\n")
                # print out the numerator
                child_counter = 0
                max_count = len(exp._numerator)-1
                for child_exp in exp._numerator:
                    if child_counter < max_count:
                        OUTPUT.write(prod_str)
                    self._print_nonlinear_terms_NL(child_exp)
                    child_counter += 1
                if denom_exists:
                    # print out the denominator
                    child_counter = 0
                    max_count = len(exp._denominator)-1
                    for child_exp in exp._denominator:
                        if child_counter < max_count:
                            OUTPUT.write(prod_str)
                        self._print_nonlinear_terms_NL(child_exp)
                        child_counter += 1

            elif _using_pyomo4_trees and (exp_type is expr._ProductExpression):
                prod_str = self._op_string[expr._ProductExpression]
                child_counter = 0
                max_count = len(exp._args)-1
                for child_exp in exp._args:
                    if child_counter < max_count:
                        OUTPUT.write(prod_str)
                    self._print_nonlinear_terms_NL(child_exp)
                    child_counter += 1

            elif _using_pyomo4_trees and (exp_type is expr._DivisionExpression):
                div_str = self._op_string[expr._DivisionExpression]
                child_counter = 0
                max_count = len(exp._args)-1
                for child_exp in exp._args:
                    if child_counter < max_count:
                        OUTPUT.write(div_str)
                    self._print_nonlinear_terms_NL(child_exp)
                    child_counter += 1

            elif _using_pyomo4_trees and (exp_type is expr._NegationExpression):
                assert len(exp._args) == 1
                OUTPUT.write(self._op_string[expr._NegationExpression])
                self._print_nonlinear_terms_NL(exp._args[0])

            elif exp_type is expr._ExternalFunctionExpression:
                fun_str, string_arg_str = \
                    self._op_string[expr._ExternalFunctionExpression]
                if not self._symbolic_solver_labels:
                    OUTPUT.write(fun_str
                                 % (self.external_byFcn[exp._fcn._function][1],
                                    len(exp._args)))
                else:
                    # Note: exp.name fails
                    OUTPUT.write(fun_str
                                 % (self.external_byFcn[exp._fcn._function][1],
                                    len(exp._args),
                                    exp.name))
                for arg in exp._args:
                    if isinstance(arg, basestring):
                        OUTPUT.write(string_arg_str % (len(arg), arg))
                    else:
                        self._print_nonlinear_terms_NL(arg)
            elif (exp_type is expr._PowExpression) or \
                 isinstance(exp, expr._IntrinsicFunctionExpression):
                intr_expr_str = self._op_string.get(exp.name)
                if intr_expr_str is not None:
                    OUTPUT.write(intr_expr_str)
                else:
                    logger.error("Unsupported intrinsic function ({0})",
                                 exp.name)
                    raise TypeError("ASL writer does not support '%s' expressions"
                                    % (exp.name))

                for child_exp in exp._args:
                    self._print_nonlinear_terms_NL(child_exp)
            elif exp_type is expr.Expr_if:
                OUTPUT.write(self._op_string[expr.Expr_if])
                self._print_nonlinear_terms_NL(exp._if)
                self._print_nonlinear_terms_NL(exp._then)
                self._print_nonlinear_terms_NL(exp._else)
            elif exp_type is expr._InequalityExpression:
                and_str, lt_str, le_str = \
                    self._op_string[expr._InequalityExpression]
                len_args = len(exp._args)
                assert len_args in [2,3]
                left = exp._args[0]
                middle = exp._args[1]
                right = None
                if len_args == 3:
                    right = exp._args[2]
                    OUTPUT.write(and_str)
                if exp._strict[0]:
                    OUTPUT.write(lt_str)
                else:
                    OUTPUT.write(le_str)
                self._print_nonlinear_terms_NL(left)
                self._print_nonlinear_terms_NL(middle)
                if not right is None:
                    if exp._strict[1]:
                        OUTPUT.write(lt_str)
                    else:
                        OUTPUT.write(le_str)
                    self._print_nonlinear_terms_NL(middle)
                    self._print_nonlinear_terms_NL(right)
            elif exp_type is expr._EqualityExpression:
                OUTPUT.write(self._op_string[expr._EqualityExpression])
                self._print_nonlinear_terms_NL(exp._args[0])
                self._print_nonlinear_terms_NL(exp._args[1])
            elif isinstance(exp, _ExpressionData):
                self._print_nonlinear_terms_NL(exp.expr)
            else:
                raise ValueError(
                    "Unsupported expression type (%s) in _print_nonlinear_terms_NL"
                    % (exp_type))

        elif isinstance(exp,var._VarData) and (not exp.is_fixed()):
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
            os.environ["PYOMO_AMPLFUNC"] = "\n".join(sorted(external_Libs))
        elif "PYOMO_AMPLFUNC" in os.environ:
            del os.environ["PYOMO_AMPLFUNC"]

        subsection_timer.reset()

        # Cache the list of model blocks so we don't have to call
        # model.block_data_objects() many many times
        all_blocks_list = list(model.block_data_objects(active=True, sort=sorter))

        # create a deterministic var labeling
        cntr = 0
        for block in all_blocks_list:
            vars_counter = tuple(enumerate(
                block.component_data_objects(Var,
                                             active=True,
                                             sort=sorter,
                                             descend_into=False),
                cntr))
            cntr += len(vars_counter)
            Vars_dict.update(vars_counter)
        self._varID_map = dict((id(val),key) for key,val in iteritems(Vars_dict))
        self_varID_map = self._varID_map
        # Use to label the rest of the components (which we will not encounter twice)
        trivial_labeler = _Counter(cntr)

        #
        # Count number of objectives and build the ampl_repns
        #
        n_objs = 0
        n_nonlinear_objs = 0
        ObjVars = set()
        ObjNonlinearVars = set()
        ObjNonlinearVarsInt = set()
        for block in all_blocks_list:

            gen_obj_ampl_repn = \
                getattr(block, "_gen_obj_ampl_repn", True)

            # Get/Create the ComponentMap for the repn
            if not hasattr(block,'_ampl_repn'):
                block._ampl_repn = ComponentMap()
            block_ampl_repn = block._ampl_repn

            for active_objective in block.component_data_objects(Objective,
                                                                 active=True,
                                                                 sort=sorter,
                                                                 descend_into=False):
                if symbolic_solver_labels:
                    objname = name_labeler(active_objective)
                    if len(objname) > max_rowname_len:
                        max_rowname_len = len(objname)

                if gen_obj_ampl_repn:
                    ampl_repn = generate_ampl_repn(active_objective.expr)
                    block_ampl_repn[active_objective] = ampl_repn
                else:
                    ampl_repn = block_ampl_repn[active_objective]

                wrapped_ampl_repn = RepnWrapper(
                    ampl_repn,
                    list(self_varID_map[id(var)] for var in ampl_repn._linear_vars),
                    list(self_varID_map[id(var)] for var in ampl_repn._nonlinear_vars))

                LinearVars.update(wrapped_ampl_repn._linear_vars)
                ObjNonlinearVars.update(wrapped_ampl_repn._nonlinear_vars)

                ObjVars.update(wrapped_ampl_repn._linear_vars)
                ObjVars.update(wrapped_ampl_repn._nonlinear_vars)

                obj_ID = trivial_labeler(active_objective)
                Objectives_dict[obj_ID] = (active_objective, wrapped_ampl_repn)
                self_ampl_obj_id[obj_ID] = n_objs
                symbol_map.addSymbols([(active_objective, "o%d"%n_objs)])

                n_objs += 1
                if ampl_repn.is_nonlinear():
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
        # Count number of constraints and build the ampl_repns
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

            gen_con_ampl_repn = \
                getattr(block, "_gen_con_ampl_repn", True)

            # Get/Create the ComponentMap for the repn
            if not hasattr(block,'_ampl_repn'):
                block._ampl_repn = ComponentMap()
            block_ampl_repn = block._ampl_repn

            # Initializing the constraint dictionary
            for constraint_data in block.component_data_objects(Constraint,
                                                                active=True,
                                                                sort=sorter,
                                                                descend_into=False):
                if symbolic_solver_labels:
                    conname = name_labeler(constraint_data)
                    if len(conname) > max_rowname_len:
                        max_rowname_len = len(conname)

                if gen_con_ampl_repn:
                    ampl_repn = generate_ampl_repn(constraint_data.body)
                    block_ampl_repn[constraint_data] = ampl_repn
                else:
                    ampl_repn = block_ampl_repn[constraint_data]

                ### GAH: Even if this is fixed, it is still useful to
                ###      write out these types of constraints
                ###      (trivial) as a feasibility check for fixed
                ###      variables, in which case the solver will pick
                ###      up on the model infeasibility.
                if skip_trivial_constraints and ampl_repn.is_fixed():
                    continue

                con_ID = trivial_labeler(constraint_data)
                wrapped_ampl_repn = RepnWrapper(
                    ampl_repn,
                    list(self_varID_map[id(var)] for var in ampl_repn._linear_vars),
                    list(self_varID_map[id(var)] for var in ampl_repn._nonlinear_vars))

                if ampl_repn.is_nonlinear():
                    nonlin_con_order_list.append(con_ID)
                    n_nonlinear_constraints += 1
                else:
                    lin_con_order_list.append(con_ID)

                Constraints_dict[con_ID] = (constraint_data, wrapped_ampl_repn)

                LinearVars.update(wrapped_ampl_repn._linear_vars)
                ConNonlinearVars.update(wrapped_ampl_repn._nonlinear_vars)

                nnz_grad_constraints += \
                    len(set(wrapped_ampl_repn._linear_vars).union(
                        wrapped_ampl_repn._nonlinear_vars))

                L = None
                U = None
                if constraint_data.lower is not None:
                    L = self._get_bound(constraint_data.lower)
                if constraint_data.upper is not None:
                    U = self._get_bound(constraint_data.upper)

                offset = ampl_repn._constant
                #if constraint_data.equality:
                #    assert L == U
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
                    if ampl_repn.is_nonlinear():
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
                        raise ValueError(msg.format(con_ID, str(L), str(U)))
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
                LinearVars.update(self_varID_map[id(vardata)]
                                  for vardata in soscondata.get_variables())

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
            if var.is_integer():
                LinearVarsInt.add(var_ID)
            elif var.is_binary():
                L = var.lb
                U = var.ub
                if L is None or U is None:
                    raise ValueError("Variable " + str(var.name) +\
                                     "is binary, but does not have lb and ub set")
                LinearVarsBool.add(var_ID)
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

        ##################################################
        ############ # LQM additions start here ##########
        ##################################################

        ###
        # Generate file containing matrices for continuity variables between blocks.
        # This checks for a LOCAL suffix 'lqm' on variables.
        ###
        lqm_suffix = model.component("lqm")
        if (lqm_suffix is None) or \
           (not (lqm_suffix.type() is Suffix)) or \
           (not lqm_suffix.active) or \
           (not (lqm_suffix.getDirection() is Suffix.LOCAL)):
            lqm_suffix = None
        if lqm_suffix is not None:
            lqm_var_column_ids = []
            for var_ID in full_var_list:
                lqm = lqm_suffix.get(Vars_dict[var_ID])
                if lqm != -1:
                    # store the column index (and translate to one-based indexing
                    lqm_var_column_ids.append(self_ampl_var_id[var_ID]+1)

            num_lqm_vars = len(lqm_var_column_ids)
            if num_lqm_vars > 0:
                lqm_output_name = OUTPUT.name.split('.')[0]+".lqm"
                LQM_OUTPUT = open(lqm_output_name,'w')
                LQM_OUTPUT.write("Matrix Li\n")
                LQM_OUTPUT.write("%d\n"%(len(full_var_list)))
                LQM_OUTPUT.write("%d\n"%(num_lqm_vars))
                LQM_OUTPUT.write("%d\n"%(num_lqm_vars))
                # The matrix uses one based indexing
                for row_id, col_id in enumerate(lqm_var_column_ids,1):
                    LQM_OUTPUT.write("%d %d 1\n" % (row_id, col_id))
                LQM_OUTPUT.write("Matrix Qi\n")
                LQM_OUTPUT.write("%d\n" % (num_lqm_vars))
                LQM_OUTPUT.write("%d\n" % (num_lqm_vars))
                LQM_OUTPUT.write("%d\n" % (num_lqm_vars))
                # one based indexing
                for counter in xrange(1,num_lqm_vars+1):
                    LQM_OUTPUT.write("%d %d -1\n" % (counter, counter))
                LQM_OUTPUT.close()

        ##################################################
        ############ # LQM additions end here ############
        ##################################################

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
        for block in all_blocks_list:
            for name, suf in active_export_suffix_generator(block):
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
        for suffix_name, suffixes in iteritems(suffix_dict):
            datatypes = set()
            for suffix in suffixes:
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
            con_data, wrapped_ampl_repn = Constraints_dict[con_ID]
            row_id = self_ampl_con_id[con_ID]
            OUTPUT.write("C%d" % (row_id))
            if symbolic_solver_labels:
                lbl = name_labeler(con_data)
                OUTPUT.write("\t#%s" % (lbl))
                rowf.write(lbl+"\n")
            OUTPUT.write("\n")
            self._print_nonlinear_terms_NL(wrapped_ampl_repn.repn._nonlinear_expr)

            for var_ID in set(wrapped_ampl_repn._linear_vars).union(
                    wrapped_ampl_repn._nonlinear_vars):
                cu[self_ampl_var_id[var_ID]] += 1

        for con_ID in lin_con_order_list:
            con_data, wrapped_ampl_repn = Constraints_dict[con_ID]
            row_id = self_ampl_con_id[con_ID]
            con_vars = set(wrapped_ampl_repn._linear_vars)
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
        for obj_ID, (obj, wrapped_ampl_repn) in iteritems(Objectives_dict):

            k = 0
            if not obj.is_minimizing():
                k = 1

            OUTPUT.write("O%d %d" % (self_ampl_obj_id[obj_ID], k))
            if symbolic_solver_labels:
                lbl = name_labeler(obj)
                OUTPUT.write("\t#%s" % (lbl))
                rowf.write(lbl+"\n")
            OUTPUT.write("\n")

            if wrapped_ampl_repn.repn.is_linear():
                OUTPUT.write(self._op_string[NumericConstant]
                             % (wrapped_ampl_repn.repn._constant))
            else:
                if wrapped_ampl_repn.repn._constant != 0:
                    _, binary_sum_str, _ = self._op_string[expr._SumExpression]
                    OUTPUT.write(binary_sum_str)
                    OUTPUT.write(self._op_string[NumericConstant]
                                 % (wrapped_ampl_repn.repn._constant))
                self._print_nonlinear_terms_NL(wrapped_ampl_repn.repn._nonlinear_expr)

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
                L = var.value
                U = var.value
            else:
                L = var.lb
                if L == -infinity:
                    L = None
                U = var.ub
                if U == infinity:
                    U = None
            if L is not None:
                Lv = value(L)
                if U is not None:
                    Uv = value(U)
                    if Lv == Uv:
                        var_bound_list.append("4 %r\n" % (Lv))
                    else:
                        var_bound_list.append("0 %r %r\n" % (Lv, Uv))
                else:
                    var_bound_list.append("2 %r\n" % (Lv))
            elif U is not None:
                var_bound_list.append("1 %r\n" % (value(U)))
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
            con_data, wrapped_ampl_repn = Constraints_dict[con_ID]
            num_nonlinear_vars = len(wrapped_ampl_repn._nonlinear_vars)
            num_linear_vars = len(wrapped_ampl_repn._linear_vars)
            if num_nonlinear_vars == 0:
                if num_linear_vars > 0:
                    linear_dict = dict((var_ID, coef)
                                       for var_ID, coef in
                                       zip(wrapped_ampl_repn._linear_vars,
                                           wrapped_ampl_repn.repn._linear_terms_coef))
                    OUTPUT.write("J%d %d\n"%(nc, num_linear_vars))
                    OUTPUT.writelines(
                        "%d %r\n" % (self_ampl_var_id[con_var],
                                     linear_dict[con_var])
                        for con_var in sorted(linear_dict.keys()))
            elif num_linear_vars == 0:
                nl_con_vars = \
                    sorted(wrapped_ampl_repn._nonlinear_vars)
                OUTPUT.write("J%d %d\n"%(nc, num_nonlinear_vars))
                OUTPUT.writelines(
                    "%d 0\n"%(self_ampl_var_id[con_var])
                    for con_var in nl_con_vars)
            else:
                con_vars = set(wrapped_ampl_repn._nonlinear_vars)
                nl_con_vars = sorted(
                    con_vars.difference(
                        wrapped_ampl_repn._linear_vars))
                con_vars.update(wrapped_ampl_repn._linear_vars)
                linear_dict = dict(
                    (var_ID, coef) for var_ID, coef in
                    zip(wrapped_ampl_repn._linear_vars,
                        wrapped_ampl_repn.repn._linear_terms_coef))
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
        for obj_ID, (obj, wrapped_ampl_repn) in \
               iteritems(Objectives_dict):

            grad_entries = {}
            for idx, obj_var in enumerate(
                    wrapped_ampl_repn._linear_vars):
                grad_entries[self_ampl_var_id[obj_var]] = \
                    wrapped_ampl_repn.repn._linear_terms_coef[idx]
            for obj_var in wrapped_ampl_repn._nonlinear_vars:
                if obj_var not in wrapped_ampl_repn._linear_vars:
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


# Switch from Python to C generate_ampl_repn function when possible
#try:
#    py_generate_ampl_repn = generate_ampl_repn
#    from cAmpl import generate_ampl_repn
#except ImportError:
#    del py_generate_ampl_repn

# Alternative: import C implementation under another name for testing
#from cAmpl import generate_ampl_repn as cgar
#__all__.append('cgar')
