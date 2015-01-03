#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

#
# Problem Writer for BARON .bar Format Files
#

import logging

from six import iterkeys, itervalues, iteritems, advance_iterator, StringIO
from six.moves import xrange, zip

from pyomo.opt import ProblemFormat
from pyomo.opt.base import AbstractProblemWriter
from pyomo.core.base import SymbolMap, BasicSymbolMap, TextLabeler, NumericLabeler
from pyomo.core.base import BooleanSet, Constraint, ConstraintList, expr, IntegerSet, Component
#from pyomo.core import Var, value, label_from_name, NumericConstant, Suffix
from pyomo.core.base.objective import Objective

from pyomo.util._plugin import alias


import pyutilib.services

#from StringIO import StringIO #CLH: I added this to make expr.to_string() work for const. and obj writing
from pyomo.core.base import Constraint, Var, Param, Model
from pyomo.core.base.set_types import * #CLH: added this to be able to recognize variable types when initializing them for baron 
from pyomo.core.base.suffix import active_export_suffix_generator #CLH: EXPORT suffixes "constraint_types" and "branching_priorities" pass their respective information to the .bar file


logger = logging.getLogger('pyomo.core')

class ProblemWriter_bar(AbstractProblemWriter):

    alias('baron_writer')
    alias('bar')

    def __init__(self):

        AbstractProblemWriter.__init__(self, ProblemFormat.bar)

        #Coppied from cpxlp.py:
        # Keven Hunter made a nice point about using %.16g in his attachment
        # to ticket #4319. I am adjusting this to %.17g as this mocks the
        # behavior of using %r (i.e., float('%r'%<number>) == <number>) with
        # the added benefit of outputting (+/-). The only case where this
        # fails to mock the behavior of %r is for large (long) integers (L),
        # which is a rare case to run into and is probably indicative of
        # other issues with the model.
        # *** NOTE ***: If you use 'r' or 's' here, it will break code that
        #               relies on using '%+' before the formatting character
        #               and you will need to go add extra logic to output
        #               the number's sign.
        self._precision_string = '.17g'

    def __call__(self, model, output_filename, solver_capability, io_options):



        # NOTE: io_options is a simple dictionary of keyword-value pairs
        #       specific to this writer.

        symbolic_solver_labels = io_options.pop("symbolic_solver_labels", False)
        output_fixed_variable_bounds = io_options.pop("output_fixed_variable_bounds", False)
        labeler = io_options.pop("labeler", None)

        # How much effort do we want to put into ensuring the
        # LP file is written deterministically for a Pyomo model:
        #    0 : None
        #    1 : sort keys of indexed components (default)
        #    2 : sort keys AND sort names (over declaration order)
        file_determinism = io_options.pop("file_determinism", 1)

        if io_options:
            logger.warn(
                "ProblemWriter_baron_writer passed unrecognized io_options:\n\t" +
                "\n\t".join("%s = %s" % (k,v) for k,v in iteritems(io_options)))

        if symbolic_solver_labels and (labeler is not None):
            raise ValueError("ProblemWriter_baron_writer: Using both the "
                             "'symbolic_solver_labels' and 'labeler' "
                             "I/O options is forbidden")

        if symbolic_solver_labels:
            labeler = TextLabeler()
        elif labeler is None:
            labeler = NumericLabeler('x')

        if output_filename is None:
            output_filename = model.name + ".bar"

        output_file=open(output_filename, "a")

        #*******Begin section copied from BARON.py**********
        #print('*********************************************')
        #print('Executing bar writer in baron_writer.py')
        #print('*********************************************')
        
        # 
        # Populate the symbol_map
        #

        symbol_map = SymbolMap(model)
       
        #cache frequently called functions
        create_symbol_func = SymbolMap.createSymbol
        create_symbols_func = SymbolMap.createSymbols
        alias_symbol_func = SymbolMap.alias


        for block in model.all_blocks(active=True, sort=SortComponents.deterministic):
            for name,index,objective_data in block.component_data_iter(ctype=Objective, active=True):
                create_symbol_func(symbol_map, objective_data, labeler)            

        for block in model.all_blocks(active=True, sort=SortComponents.deterministic):
            for name,index,constraint_data in block.component_data_iter(ctype=Constraint, active=True):
                constraint_data_symbol = create_symbol_func(symbol_map, constraint_data, labeler)
                label = 'con_' + constraint_data_symbol
                alias_symbol_func(symbol_map, constraint_data, label)

        for block in model.all_blocks(active=True, sort=SortComponents.deterministic):
            create_symbols_func(symbol_map, block.active_components_data_iter(Var), labeler)

        for block in model.all_blocks(active=True, sort=SortComponents.deterministic):
            for name,index,param_data in block.component_data_iter(ctype=Param, active=True):
                #instead of checking if param_data._mutable:
                if not param_data.is_constant():
                    create_symbol_func(symbol_map, param_data, labeler)

        object_symbol_dictionary = symbol_map.getByObjectDictionary()


        # If the text labeler is used, corect the labels to be baron-allowed variable names
        # Change '(' and ')' to '__' 
        # This way, for simple variable names like 'x(1_2)' --> 'x__1_2__'
        # FIXME: 7/21/14  This may break if users give variable names with two or more underscores together
        if symbolic_solver_labels:
            for key,label in iteritems(object_symbol_dictionary):
                label = label.replace('(','___')
                object_symbol_dictionary[key] = label.replace(')','__')
                

        #
        # BINARY_VARIABLES, INTEGER_VARIABLES, POSITIVE_VARIABLES, VARIABLES
        #

        nbv = 0 
        niv = 0
        npv = 0
        nv = 0

        BinVars = []
        IntVars = []
        PosVars = []
        Vars = []

        for block in model.all_blocks(active=True, sort=SortComponents.deterministic):
            for name,index,variable in block.component_data_iter(ctype=Var, active=True):    
                if isinstance(variable.domain, BooleanSet):
                    nbv += 1
                    TypeList = BinVars
                elif isinstance(variable.domain, IntegerSet):
                    niv += 1
                    TypeList = IntVars
                elif isinstance(variable.domain, RealSet) and variable.lb is not None and variable.lb >= 0: 
                    npv += 1
                    TypeList = PosVars
                else:
                    nv += 1
                    TypeList = Vars
                    
                var_name = object_symbol_dictionary[id(variable)]
                TypeList.append(var_name)

        if nbv != 0:
            output_file.write('BINARY_VARIABLES ')
            for var_name in BinVars[:-1]:
                output_file.write(str(var_name)+', ')
            var_name = BinVars[nbv-1]
            output_file.write(str(var_name)+';\n\n')
        if niv != 0:
            output_file.write('INTEGER_VARIABLES ')
            for var_name in IntVars[:-1]:
                output_file.write(str(var_name)+', ')
            var_name = IntVars[niv-1]
            output_file.write(str(var_name)+';\n\n')
        if npv != 0:
            output_file.write('POSITIVE_VARIABLES ')
            for var_name in PosVars[:-1]:
                output_file.write(str(var_name)+', ')
            var_name = PosVars[npv-1]
            output_file.write(str(var_name)+';\n\n')
        if nv != 0:
            output_file.write('VARIABLES ')
            for var_name in Vars[:-1]:
                output_file.write(str(var_name)+', ')
            var_name = Vars[nv-1]
            output_file.write(str(var_name)+';\n\n')

        
        #
        # LOWER_BOUNDS
        # 

        LowerBoundHeader = False
        for block in model.all_blocks(active=True, sort=SortComponents.deterministic):
            for name,index,var_value in block.component_data_iter(ctype=Var, active=True):
                if var_value.fixed:
                    var_value_lb = var_value.value
                else:
                    var_value_lb = var_value.lb

                if var_value_lb is not None:
                    if LowerBoundHeader is False:
                        output_file.write("LOWER_BOUNDS{\n")
                        LowerBoundHeader = True
                    name_to_output = object_symbol_dictionary[id(var_value)]
                    lb_string_template = '%s: %'+self._precision_string+';\n'
                    output_file.write(lb_string_template % (name_to_output, var_value_lb)) 

        if LowerBoundHeader:
            output_file.write("}\n\n")
                    
        #
        # UPPER_BOUNDS
        # 

        UpperBoundHeader = False
        for block in model.all_blocks(active=True, sort=SortComponents.deterministic):
            for name,index,var_value in block.component_data_iter(ctype=Var, active=True):
                if var_value.fixed:
                    var_value_ub = var_value.value
                else:
                    var_value_ub = var_value.ub

                if var_value_ub is not None:
                    if UpperBoundHeader is False:
                        output_file.write("UPPER_BOUNDS{\n")
                        UpperBoundHeader = True
                    name_to_output = object_symbol_dictionary[id(var_value)]
                    ub_string_template = '%s: %'+self._precision_string+';\n'
                    output_file.write(ub_string_template % (name_to_output, var_value_ub)) 

        if UpperBoundHeader:
            output_file.write("}\n\n")

        #
        # BRANCHING_PRIORITIES
        #

        # Specifyig priorities requires that the pyomo model has established an 
        # EXTERNAL, float suffix called 'branching_priorities' on the model 
        # object, indexed by the relevant variable
        BranchingPriorityHeader = False

        for block in model.all_blocks(active=True, sort=SortComponents.deterministic):
            for name,suffix in active_export_suffix_generator(block):
                if name == 'branching_priorities':
                    for name,index,var_value in block.component_data_iter(ctype=Var, active=True):
                        priority = suffix.getValue(variable)
                        if priority is not None:
                            if not BranchingPriorityHeader:
                                output_file.write('BRANCHING_PRIORITIES{\n')
                                BranchingPriorityHeader = True
                            name_to_output = object_symbol_dictionary[id(variable)]
                            output_file.write(name_to_output+': '+str(priority)+';\n')

        if BranchingPriorityHeader:
            output_file.write("}\n\n")

        #
        # EQUATIONS
        #
        
        # Equation Counting
        eqns = []               
        r_o_eqns = []
        c_eqns = []
        l_eqns = []             
        EquationHeader = False
        for block in model.all_blocks(active=True, sort=SortComponents.deterministic):
            for name,index,constraint_data in block.component_data_iter(ctype=Constraint, active=True):

                #FIXME: CLH, 7/18/14: Not sure if the code for .trivial is up-to-date and needs to here.
                #if constraint_data.parent_component().trivial: 
                #    continue
               
                flag = False
                for name,suffix in active_export_suffix_generator(block):
                    if name == 'constraint_types':
                        constraint_type = suffix.getValue(constraint_data)
                        if constraint_type is None:
                            flag = True
                            eqns.append(constraint_data)
                        elif constraint_type.lower() == 'relaxationonly':
                            flag = True
                            r_o_eqns.append(constraint_data)
                        elif constraint_type.lower() == 'convex':
                            flag = True
                            c_eqns.append(constraint_data)
                        elif constraint_type.lower() == 'local':
                            flag = True
                            l_eqns.append(constraint_data)
                        else:
                            logger.warn('A constraint_types suffix was not recognized: %s' % constraint_type)
                if not flag:
                    eqns.append(constraint_data)


        #Equation Declaration
        n_eqns = len(eqns)
        n_roeqns = len(r_o_eqns)
        n_ceqns = len(c_eqns)
        n_leqns = len(l_eqns)

        if n_eqns > 0:
            output_file.write('EQUATIONS ')
            for i in xrange(n_eqns):
                eqn = eqns[i]
                con_symbol = object_symbol_dictionary[id(eqn)]
                if i == n_eqns-1:
                    output_file.write(str(con_symbol)+';\n\n')
                else:
                    output_file.write(str(con_symbol)+', ')              

        if n_roeqns > 0:
            output_file.write('RELAXATION_ONLY_EQUATIONS ')
            for i in xrange(n_roeqns):
                eqn = r_o_eqns[i]
                con_symbol = object_symbol_dictionary[id(eqn)]
                if i == n_roeqns-1:
                    output_file.write(str(con_symbol)+';\n\n')
                else:
                    output_file.write(str(con_symbol)+', ')              

        if n_ceqns > 0:
            output_file.write('CONVEX_EQUATIONS ')
            for i in xrange(n_ceqns):
                eqn = c_eqns[i]
                con_symbol = object_symbol_dictionary[id(eqn)]
                if i == n_ceqns-1:
                    output_file.write(str(con_symbol)+';\n\n')
                else:
                    output_file.write(str(con_symbol)+', ')              

        if n_leqns > 0:
            output_file.write('LOCAL_EQUATIONS ')
            for i in xrange(n_leqns):
                eqn = l_eqns[i]
                con_symbol = object_symbol_dictionary[id(eqn)]
                if i == n_leqns-1:
                    output_file.write(str(con_symbol)+';\n\n')
                else:
                    output_file.write(str(con_symbol)+', ')              


        # Create a dictionary of baron variable names to mach to the strings that 
        # constraint.to_string() prints. An important note is that the variable strings 
        # are padded by spaces so that whole variable names are recognized, and simple 
        # variable names are not identified inside loger names. 
        # Example: ' x[1] ' -> ' x3 '
        #FIXME: 7/18/14 CLH: This may cause mistakes if spaces in variable names are allowed
        string_to_bar_dict = {}
        for block in model.all_blocks(active=True, sort=SortComponents.deterministic):
            for name,index,variable in block.component_data_iter(ctype=Var, active=True):
                variable_stream = StringIO()
                variable.to_string(ostream=variable_stream, verbose=False)
                variable_string = variable_stream.getvalue()
                
                variable_string = ' '+variable_string+' '
                string_to_bar_dict[variable_string] = ' '+object_symbol_dictionary[id(variable)]+' '
                   
            for name,index,param in block.component_data_iter(ctype=Param, active=True):
                #if param._mutable:
                if param.is_constant():
                    param_stream = StringIO()
                    param.to_string(ostream=param_stream, verbose=False)
                    param_string = param_stream.getvalue()
                    
                    param_string = ' '+param_string+' '
                    string_to_bar_dict[param_string] = ' '+str(param.value)+' '
                
        

        # Equation Definition
        for block in model.all_blocks(active=True, sort=SortComponents.deterministic):
            #for constraint in active_components_generator(block, Constraint):
            for name,index,constraint_data in block.component_data_iter(ctype=Constraint, active=True):

                #FIXME: 7/18/14 CLH: same as above, not sure if .trivial is necessary anymore
                #if constraint_data.parent_component().trivial:
                #    continue

                if not constraint_data.active: 
                    continue
         
                con_symbol = object_symbol_dictionary[id(constraint_data)]
                label = str(con_symbol) + ': '
                output_file.write(label)
                    

                #########################
                #CLH: The section below is kind of a hack-y way to use the expr.to_string
                #     function to print expressions. A stream is created, writen to, and then
                #     the string is recovered and stored in eqn_body. Then the variable names are 
                #     converted to match the variable names that are used in the bar file.

                # Fill in the body of the equation
                body_string_buffer = StringIO()
                
                constraint_data.body.to_string(ostream=body_string_buffer, verbose=False)
                eqn_body = body_string_buffer.getvalue()
                

                # First, pad the equation so that if there is a variable name at the start or
                # end of the equation, it can still be identified as padded with spaces.
                # Second, change pyomo's ** to baron's ^, also with padding so that variable
                # can always be found with space around them
                # Third, add more padding around multiplication. Pyomo already has spaces 
                # between variable on variable multiplication, but not for constants on variables
                eqn_body = ' '+eqn_body+' '
                eqn_body = eqn_body.replace('**',' ^ ')
                eqn_body = eqn_body.replace('*', ' * ')

                for variable_string in string_to_bar_dict.iterkeys():                       
                    eqn_body = eqn_body.replace(variable_string,string_to_bar_dict[variable_string])
                    

                #FIXME: 7/29/14 CLH: Baron doesn't handle many of the intrinsic_functions available 
                #                    in pyomo. The error message given by baron is also very weak.
                #                    Either a function here to re-write unallowed expressions or
                #                    a way to track solver capability by intrinsic_expression would
                #                    be useful.
                #
                ##########################


                string_template = '%'+self._precision_string

                # Fill in the left and right hand side (constants) of the equations
                # Equality constraint
                if constraint_data.equality:
                    eqn_lhs = ''                       
                    eqn_rhs = ' == ' + str(string_template % constraint_data.upper())

                # Greater than constraint
                elif constraint_data.upper is None:
                    eqn_rhs = ' >= ' + str(string_template % constraint_data.lower())
                    eqn_lhs = ''

                # Less than constraint
                elif constraint_data.lower is None:
                    eqn_rhs = ' <= ' + str(string_template % constraint_data.upper())
                    eqn_lhs = ''

                # Double-sided constraint 
                elif constraint_data.upper is not None and constraint_data.lower is not None:
                    eqn_lhs = str(string_template % constraint_data.lower()) + ' <= '
                    eqn_rhs = ' <= ' + str(string_template % constraint_data.upper())
                  
                eqn_string = eqn_lhs + eqn_body + eqn_rhs + ';\n'
                output_file.write(eqn_string)

                    
        #
        # OBJECTIVE
        #

        output_file.write("\nOBJ: ")

        for block in model.all_blocks(active=True, sort=SortComponents.deterministic):
            for name,index,objective in block.component_data_iter(ctype=Objective, active=True):

                if objective.is_minimizing():
                    output_file.write("minimize ")
                else: 
                    output_file.write("maximize ")
                
                #FIXME 7/18/14 See above, constraint writing section. Will cause problems if
                #              there are spaces in variables
                # Similar to the constraints section above, the objective is generated
                # from the expr.to_string function. 
                obj_stream = StringIO()
                objective.expr.to_string(ostream=obj_stream, verbose=False)

                obj_string = ' '+obj_stream.getvalue()+' '                               
                obj_string = obj_string.replace('**',' ^ ')
                obj_string = obj_string.replace('*', ' * ')

                for variable_string in string_to_bar_dict.iterkeys():
                    obj_string = obj_string.replace(variable_string, string_to_bar_dict[variable_string])

        output_file.write(obj_string+";\n\n")

        #
        # STARTING_POINT
        #
        starting_point_list = []
        for block in model.all_blocks(active=True, sort=SortComponents.deterministic):
            for name,index,variable in block.component_data_iter(ctype=Var, active=True):
                starting_point = variable.value
                if starting_point is not None:
                    starting_point_list.append((variable,starting_point))

        if len(starting_point_list) > 0:
            output_file.write('STARTING_POINT{\n')
            for variable,starting_value in starting_point_list:
                var_name = object_symbol_dictionary[id(variable)]
                string_template = '%s: %'+self._precision_string+';\n'
                output_file.write(string_template % (var_name, starting_value))
            output_file.write('}\n\n')


        #*******End section copied from BARON.py************
                     
        output_file.close()

        return output_filename, symbol_map

