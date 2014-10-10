#  _________________________________________________________________________
#
#  Pyomo: A COmmon Optimization Python Repository
#  Copyright (c) 2010 Sandia Corporation.
#  This software is distributed under the BSD License.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  For more information, see the Pyomo README.txt file.
#  _________________________________________________________________________


import logging
import os

from six import itervalues, iterkeys, iteritems
from six.moves import xrange, StringIO

logger = logging.getLogger('pyomo.solvers')

import pyutilib.services
import pyutilib.common
from pyutilib.misc import Bunch, Options

import pyomo.misc.plugin
from pyomo.opt.base import *
from pyomo.opt.base.solvers import _extract_version
from pyomo.opt.results import *
from pyomo.opt.solver import *
from pyomo.core.base import SymbolMap, BasicSymbolMap, NumericLabeler, TextLabeler
from pyomo.core.base.numvalue import value
from pyomo.core.base.block import active_components, active_components_data

from pyomo.core.base.objective import Objective 
from pyomo.core.base import Constraint, Var
from pyomo.core.base.set_types import * #CLH: added this to be able to recognize variable types when initializing them for baron
from pyomo.core.base.suffix import active_export_suffix_generator #CLH: EXPORT suffixes "constraint_types" and "branching_priorities" pass their respective information to the .bar file
import re #CLH: added to match the suffixes while processing the solution. Same as CPLEX.py
import tempfile #CLH: added to make a tempfile used to get the version of the baron executable. 

class BARONSHELL(SystemCallSolver):
    """The BARON MINLP solver
    """

    pyomo.misc.plugin.alias('baron',  doc='The BARON MINLP solver')

    def __init__(self, **kwds):
        #
        # Call base class constructor
        #
        kwds['type'] = 'baron'
        SystemCallSolver.__init__(self, **kwds)

        self._valid_problem_formats=[ProblemFormat.bar]
        self._valid_result_formats = {}
        self._valid_result_formats[ProblemFormat.bar] = [ResultsFormat.soln]
        self.set_problem_format(ProblemFormat.bar)

        self._capabilities = Options()
        self._capabilities.linear = True
        self._capabilities.quadratic_objective = True
        self._capabilities.quadratic_constraint = True
        self._capabilities.integer = True
        self._capabilities.sos1 = False
        self._capabilities.sos2 = False


        # CLH: Coppied from cpxlp.py, the cplex file writer. 
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

    def executable(self):
        executable = pyutilib.services.registered_executable("baron")
        if executable is None:
            logger.warning("Could not locate the 'baron' executable, "
                           "which is required for solver %s" % self.name)
            self.enable = False
            return None
        return executable.get_path()

    def version(self):
        """
        Returns a tuple describing the solver executable version.
        """
        solver_exec = self.executable()

        if solver_exec is None:
            return _extract_version('')
        else:
            dummy_prob_file = tempfile.NamedTemporaryFile()
            dummy_prob_file.write('//This is a dummy .bar file created to return the baron '+\
                                  'version//\nPOSITIVE_VARIABLES x1;\nOBJ: minimize x1;')
            dummy_prob_file.seek(0)
            results = pyutilib.subprocess.run( [solver_exec,dummy_prob_file.name])
            return _extract_version(results[1],length=3)

    def create_command_line(self, executable, problem_files):

        #
        # Define log file
        #
        log_file = pyutilib.services.TempfileManager.create_tempfile(suffix = '.baron.log')

        #
        # Define solution file
        #
        
        #CLH: I created the solution file in the _convert_problem function.
        #     The bar file needs the solution filename in the OPTIONS section, but
        #     this function is executed after the bar problem file writing.
        #self.soln_file = pyutilib.services.TempfileManager.create_tempfile(suffix = '.baron.sol')


        cmd = [executable, problem_files[0]]
        if self._timer:
            cmd.insert(0, self._timer)
        return pyutilib.misc.Bunch( cmd=cmd, 
                                    log_file=log_file,
                                    env=None )
    
    #
    # Assuming the variable values stored in the model will
    # automatically be included in the Baron input file
    # (returning True implies the opposite and requires another function)
    def warm_start_capable(self):

        return False

    def _convert_problem(self, args, problem_format, valid_problem_formats):
        
        # TODO (later):
        # self.output_fixed_variable_bounds
        # self.has_capability

        if self.symbolic_solver_labels:
            labeler = TextLabeler()
        else:
            labeler = NumericLabeler('x')    


        problem_filename = pyutilib.services.TempfileManager.create_tempfile(suffix = '.pyomo.bar')
        self.soln_file = pyutilib.services.TempfileManager.create_tempfile(suffix = '.baron.sol')
        self.tim_file = pyutilib.services.TempfileManager.create_tempfile(suffix = '.baron.tim')

        assert len(args) == 1
        instance = args[0]

        # 
        # Populate the symbol_map
        #

        symbol_map = SymbolMap(instance)
       
        #cache frequently called functions
        create_symbol_func = SymbolMap.createSymbol
        create_symbols_func = SymbolMap.createSymbols
        alias_symbol_func = SymbolMap.alias


        for block in instance.all_blocks(sort_by_keys=True):
            for name,index,objective_data in block.component_data(ctype=Objective, active=True):
                create_symbol_func(symbol_map, objective_data, labeler)            

        for block in instance.all_blocks(sort_by_keys=True):
            for name,index,constraint_data in block.component_data(ctype=Constraint, active=True):
                constraint_data_symbol = create_symbol_func(symbol_map, constraint_data, labeler)
                label = 'con_' + constraint_data_symbol
                alias_symbol_func(symbol_map, constraint_data, label)

        create_symbols_func(symbol_map, active_components_data(block, Var), labeler)

        object_symbol_dictionary = symbol_map.getByObjectDictionary()

        #CLH: I added this to have acces to the symbol map in the process_soln function
        self._symbol_map = symbol_map


        # If the text labeler is used, corect the labels to be baron-allowed variable names
        # Change '(' and ')' to '__' 
        # This way, for simple variable names like 'x(1_2)' --> 'x__1_2__'
        # FIXME: 7/21/14  This may break if users give variable names with two or more underscores together
        if self.symbolic_solver_labels:
            for key,label in iteritems(object_symbol_dictionary):
                label = label.replace('(','___')
                object_symbol_dictionary[key] = label.replace(')','__')
                

        #
        # Begin writing the .bar file
        #

        prob_file = open(problem_filename,'r+')
        prob_file.write("// Source Pyomo model name="+str(instance.name)+"\n\n")

        #
        # OPTIONS
        #
        prob_file.write("OPTIONS{\nResName: \""+self.soln_file+"\";\n")


        ##################
        #
        # CLH: Baron outputs a warning line:
        #      "BARON option ### not recognized. Option will be ignored."
        #      or:
        #      "Supplied value of ### is <below/above> allowable <minimum/maximum>
        #       Setting ### to  ###  and continuing."
        #      So, checking for problematic options may not be necessary. 
        #
        #Baron_Options = ['epsa','epsr','deltaterm','deltat','deltaa','deltar','cutoff',
        #                 'absconfeastol','relconfeastol','absintfeastol','relintfeastol',
        #                 'boxtol','firstfeas','maxiter','maxtime','numsol','isoltol',
        #                 'nouter1','noutpervar','noutiter','outgrid','tdo','mdo','lbttdo',
        #                 'obttdo','pdo','brvarstra','brptstra','nodesel','bpint','dolocal',
        #                 'numloc','prfreq','prtimefreq','prlevel','locres','proname',
        #                 'results','resname','summary','sumname','times','timname','lpsol',
        #                 'lpalg','nlpsol','allowminos','allowsnopt','allowipopt','licname']
        #
        #
        #for key in self.options:
        #    lower_key = key.lower()
        #    if lower_key in Baron_Options:
        #        if lower_key == 'resname':
        #            pass
        #        elif lower_key == 'timname':
        #            pass
        #        elif lower_key == 'sumname':
        #            pass
        #        else:
        #            prob_file.write(key+": "+str(self.options[key])+";\n")
        #    else:
        #        #CLH: A warning or error needs to be raised here to indicate that an 
        #        #     invalid option was passed to pyomo. I wasn't sure the correct level
        #        #     to implement though
        #        logger.warn('An option passed in --solver-options was not a valid BARON option')
        #
        #######################


        # Process the --solver-options options. Rely on baron to catch and reset bad option values
        sum_flag = False
        for key in self.options:
            lower_key = key.lower()
            if lower_key == 'resname':
                logger.warn('The resname option is set to %s' % self.soln_file)
            elif lower_key == 'timname':
                logger.warn('The timname option is set to %s' % self.tim_file)
            else:
                if lower_key == 'summary' and self.options[key] == 1:
                    sum_flag = True
                prob_file.write(key+": "+str(self.options[key])+";\n")

        # The 'summary option is defaulted to 0, so that no summary file is generated
        # in the directory where the user calls baron. Check if a user explicitly asked
        # for a summary file. 
        if sum_flag == True:
            prob_file.write("Summary: 1;\n")
        else:
            prob_file.write("Summary: 0;\n")

        prob_file.write("TimName: \""+self.tim_file+"\";\n}\n\n")
      
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


        for block in instance.all_blocks(sort_by_keys=True):
            for name,index,variable in block.component_data(ctype=Var, active=True):    
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
            prob_file.write('BINARY_VARIABLES ')
            for var_name in BinVars[:-1]:
                prob_file.write(str(var_name)+', ')
            var_name = BinVars[nbv-1]
            prob_file.write(str(var_name)+';\n\n')
        if niv != 0:
            prob_file.write('INTEGER_VARIABLES ')
            for var_name in IntVars[:-1]:
                prob_file.write(str(var_name)+', ')
            var_name = IntVars[niv-1]
            prob_file.write(str(var_name)+';\n\n')
        if npv != 0:
            prob_file.write('POSITIVE_VARIABLES ')
            for var_name in PosVars[:-1]:
                prob_file.write(str(var_name)+', ')
            var_name = PosVars[npv-1]
            prob_file.write(str(var_name)+';\n\n')
        if nv != 0:
            prob_file.write('VARIABLES ')
            for var_name in Vars[:-1]:
                prob_file.write(str(var_name)+', ')
            var_name = Vars[nv-1]
            prob_file.write(str(var_name)+';\n\n')

        
        #
        # LOWER_BOUNDS
        # 

        LowerBoundHeader = False
        for block in instance.all_blocks(sort_by_keys=True):
            for name,index,var_value in block.component_data(ctype=Var, active=True):
                if var_value.fixed:
                    var_value_lb = var_value.value
                else:
                    var_value_lb = var_value.lb

                if var_value_lb is not None:
                    if LowerBoundHeader is False:
                        prob_file.write("LOWER_BOUNDS{\n")
                        LowerBoundHeader = True
                    name_to_output = object_symbol_dictionary[id(var_value)]
                    lb_string_template = '%s: %'+self._precision_string+';\n'
                    prob_file.write(lb_string_template % (name_to_output, var_value_lb)) 

        if LowerBoundHeader:
            prob_file.write("}\n\n")
                    
        #
        # UPPER_BOUNDS
        # 

        UpperBoundHeader = False
        for block in instance.all_blocks(sort_by_keys=True):
            for name,index,var_value in block.component_data(ctype=Var, active=True):
                if var_value.fixed:
                    var_value_ub = var_value.value
                else:
                    var_value_ub = var_value.ub

                if var_value_ub is not None:
                    if UpperBoundHeader is False:
                        prob_file.write("UPPER_BOUNDS{\n")
                        UpperBoundHeader = True
                    name_to_output = object_symbol_dictionary[id(var_value)]
                    ub_string_template = '%s: %'+self._precision_string+';\n'
                    prob_file.write(ub_string_template % (name_to_output, var_value_ub)) 

        if UpperBoundHeader:
            prob_file.write("}\n\n")

        #
        # BRANCHING_PRIORITIES
        #

        # Specifyig priorities requires that the pyomo model has established an 
        # EXTERNAL, float suffix called 'branching_priorities' on the model 
        # object, indexed by the relevant variable
        BranchingPriorityHeader = False

        for block in instance.all_blocks(sort_by_keys=True):
            for name,suffix in active_export_suffix_generator(block):
                if name == 'branching_priorities':
                    for name,index,var_value in block.component_data(ctype=Var, active=True):
                        priority = suffix.get(variable)
                        if priority is not None:
                            if not BranchingPriorityHeader:
                                prob_file.write('BRANCHING_PRIORITIES{\n')
                                BranchingPriorityHeader = True
                            name_to_output = object_symbol_dictionary[id(variable)]
                            prob_file.write(name_to_output+': '+str(priority)+';\n')

        if BranchingPriorityHeader:
            prob_file.write("}\n\n")

        #
        # EQUATIONS
        #
        
        # Equation Counting
        eqns = []               
        r_o_eqns = []
        c_eqns = []
        l_eqns = []             
        EquationHeader = False
        for block in instance.all_blocks(sort_by_keys=True):
            for name,index,constraint_data in block.component_data(ctype=Constraint, active=True):

                #FIXME: CLH, 7/18/14: Not sure if the code for .trivial is up-to-date and needs to here.
                #if constraint_data.component().trivial: 
                #    continue
               
                flag = False
                for name,suffix in active_export_suffix_generator(block):
                    if name == 'constraint_types':
                        constraint_type = suffix.get(constraint_data)
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
            prob_file.write('EQUATIONS ')
            for i in xrange(n_eqns):
                eqn = eqns[i]
                con_symbol = object_symbol_dictionary[id(eqn)]
                if i == n_eqns-1:
                    prob_file.write(str(con_symbol)+';\n\n')
                else:
                    prob_file.write(str(con_symbol)+', ')              

        if n_roeqns > 0:
            prob_file.write('RELAXATION_ONLY_EQUATIONS ')
            for i in xrange(n_roeqns):
                eqn = r_o_eqns[i]
                con_symbol = object_symbol_dictionary[id(eqn)]
                if i == n_roeqns-1:
                    prob_file.write(str(con_symbol)+';\n\n')
                else:
                    prob_file.write(str(con_symbol)+', ')              

        if n_ceqns > 0:
            prob_file.write('CONVEX_EQUATIONS ')
            for i in xrange(n_ceqns):
                eqn = c_eqns[i]
                con_symbol = object_symbol_dictionary[id(eqn)]
                if i == n_ceqns-1:
                    prob_file.write(str(con_symbol)+';\n\n')
                else:
                    prob_file.write(str(con_symbol)+', ')              

        if n_leqns > 0:
            prob_file.write('LOCAL_EQUATIONS ')
            for i in xrange(n_leqns):
                eqn = l_eqns[i]
                con_symbol = object_symbol_dictionary[id(eqn)]
                if i == n_leqns-1:
                    prob_file.write(str(con_symbol)+';\n\n')
                else:
                    prob_file.write(str(con_symbol)+', ')              


        # Create a dictionary of baron variable names to mach to the strings that 
        # constraint.to_string() prints. An important note is that the variable strings 
        # are padded by spaces so that whole variable names are recognized, and simple 
        # variable names arent identified inside loger names. 
        # Example: ' x[1] ' -> ' x3 '
        #FIXME: 7/18/14 CLH: This may cause mistakes if spaces in variable names are allowed
        string_to_bar_dict = {}
        for block in instance.all_blocks(sort_by_keys=True):
            for name,index,variable in block.component_data(ctype=Var, active=True):
                variable_stream = StringIO()
                variable.to_string(ostream=variable_stream, verbose=False)
                variable_string = ' '+variable_stream.getvalue()+' '
                string_to_bar_dict[variable_string] = ' '+object_symbol_dictionary[id(variable)]+' '
                     
            
        # Equation Definition
        for block in instance.all_blocks(sort_by_keys=True):
            #for constraint in active_components_generator(block, Constraint):
            for name,index,constraint_data in block.component_data(ctype=Constraint, active=True):

                #FIXME: 7/18/14 CLH: same as above, not sure if .trivial is necessary anymore
                #if constraint_data.component().trivial:
                #    continue

                if not constraint_data.active: 
                    continue
         
                con_symbol = object_symbol_dictionary[id(constraint_data)]
                label = str(con_symbol) + ': '
                prob_file.write(label)
                    

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
                prob_file.write(eqn_string)

                    
        #
        # OBJECTIVE
        #

        prob_file.write("\nOBJ: ")

        for block in instance.all_blocks(sort_by_keys=True):
            for name,index,objective in block.component_data(ctype=Objective, active=True):

                if objective.is_minimizing():
                    prob_file.write("minimize ")
                else: 
                    prob_file.write("maximize ")
                
                #FIXME 7/18/14 See above, constraint writing section. Will cause problems if
                #              there are spaces in variables
                #CLH: Similar to the constraints section above, the objective is generated
                #     from the expr.to_string function. 
                obj_stream = StringIO()
                objective.expr.to_string(ostream=obj_stream, verbose=False)

                obj_string = ' '+obj_stream.getvalue()+' '                               
                obj_string = obj_string.replace('**',' ^ ')
                obj_string = obj_string.replace('*', ' * ')

                for variable_string in string_to_bar_dict.iterkeys():
                    obj_string = obj_string.replace(variable_string, string_to_bar_dict[variable_string])

        prob_file.write(obj_string+";\n\n")

        #
        # STARTING_POINT
        #
        starting_point_list = []
        for block in instance.all_blocks(sort_by_keys=True):
            for name,index,variable in block.component_data(ctype=Var, active=True):
                starting_point = variable.value
                if starting_point is not None:
                    starting_point_list.append((variable,starting_point))

        if len(starting_point_list) > 0:
            prob_file.write('STARTING_POINT{\n')
            for variable,starting_value in starting_point_list:
                var_name = object_symbol_dictionary[id(variable)]
                string_template = '%s: %'+self._precision_string+';\n'
                prob_file.write(string_template % (var_name, starting_value))
            prob_file.write('}\n\n')


#        print('\n***************Printing %s************' % problem_filename)
#        prob_file.seek(0,0)
#        print(prob_file.read())
#        print('************************************************************\n')
 

        
        return [problem_filename], ProblemFormat.bar, symbol_map


    def process_logfile(self):

        results = SolverResults()

        #
        # Process logfile
        #
        OUTPUT = open(self.log_file)

        # Collect cut-generation statistics from the log file
        for line in OUTPUT:
            if 'Bilinear' in line:
                results.solver.statistics['Bilinear_cuts'] = int(line.split()[1])
            elif 'LD-Envelopes' in line:
                results.solver.statistics['LD-Envelopes_cuts'] = int(line.split()[1])
            elif 'Multilinears' in line:
                results.solver.statistics['Multilinears_cuts'] = int(line.split()[1])
            elif 'Convexity' in line:
                results.solver.statistics['Convexity_cuts'] = int(line.split()[1])
            elif 'Integrality' in line:
                results.solver.statistics['Integrality_cuts'] = int(line.split()[1])

        OUTPUT.close()        
        return results


    def process_soln_file(self, results):

        # TODO: Is there a way to hanle non-zero return values from baron? 
        #       Example: the "NonLinearity Error if POW expression" 
        #       (caused by  x ^ y) when both x and y are variables
        #       causes an ugly python error and the solver log has a single
        #       line to display the error, hard to pick out of the list  

        # Check for suffixes to send back to pyomo
        extract_marginals = False
        extract_price = False
        for suffix in self.suffixes:
            flag = False
            if re.match(suffix, "marginal"): #baron_marginal
                extract_marginals = True
                flag = True
            if re.match(suffix, "price"): #baron_price
                extract_price = True
                flag = True
            if not flag:
                raise RuntimeError("***The BARON solver plugin cannot"
                                   "extract solution suffix="+suffix)

        # check for existence of the solution and time file. Not sure why we
        # just return - would think that we would want to indicate
        # some sort of error
        if not os.path.exists(self.soln_file):
            logger.warn("Solution file does not exist: %s" % (self.soln_file))
            return
        if not os.path.exists(self.tim_file):
            logger.warn("Time file does not exist: %s" % (self.tim_file))
            return


        symbol_map = self._symbol_map
        symbol_map_byObjects = symbol_map.byObject

        soln = Solution()

        #
        # Process model and solver status from the Baron tim file
        #

        TimFile = open(self.tim_file,"r")
        INPUT = open(self.soln_file,"r")

        line = TimFile.readline().split()
        results.problem.name = line[0]
        results.problem.number_of_constraints = int(line[1])
        results.problem.number_of_variables = int(line[2])
        results.problem.lower_bound = float(line[5])
        results.problem.upper_bound = float(line[6])
        soln.gap = results.problem.upper_bound - results.problem.lower_bound 
        solver_status = line[7]
        model_status = line[8]
  

        for name,index,obj in symbol_map.instance().component_data(ctype=Objective,active=True):
            objective_label = symbol_map_byObjects[id(obj)]
            soln.objective[objective_label].value=None
            results.problem.number_of_objectives = 1 
            results.problem.sense = 'minimizing' if obj.is_minimizing() else 'maximizing' 


        if solver_status == '1':
            results.solver.status = SolverStatus.ok
        elif solver_status == '2':
            results.solver.status = SolverStatus.error
            results.solver.termination_condition = TerminationCondition.error
            #CLH: I wasn't sure if this was double reporting errors. I just filled in one 
            #     termination_message for now
            results.solver.termination_message = 'Insufficient memory to store the number of nodes required '+\
                                                 'for this seach tree. Increase physical memory or change '+\
                                                 'algorithmic options'
        elif solver_status == '3':
            results.solver.status = SolverStatus.ok
            results.solver.termination_condition = TerminationCondition.maxIterations
        elif solver_status == '4':
            results.solver.status = SolverStatus.ok
            results.solver.termination_condition = TerminationCondition.maxTimeLimit
        elif solver_status == '5':
            results.solver.status = SolverStatus.warning
            results.solver.termination_condition = TerminationCondition.other
        elif solver_status == '6':
            results.solver.status = SolverStatus.aborted
            results.solver.termination_condition = TerminationCondition.userInterrupt
        elif solver_status == '7':
            results.solver.status = SolverStatus.error
            results.solver.termination_condition = TerminationCondition.error
        elif solver_status == '8':
            results.solver.status = SolverStatus.unknown
            results.solver.termination_condition = TerminationCondition.unknown
        elif solver_status == '9':
            results.solver.status = SolverStatus.error
            results.solver.termination_condition = TerminationCondition.solverFailure
        elif solver_status == '10':
            results.solver.status = SolverStatus.error
            results.solver.termination_condition = TerminationCondition.error
        elif solver_status == '11':
            results.solver.status = SolverStatus.aborted
            results.solver.termination_condition = TerminationCondition.licensingProblems
            results.solver.termination_message = 'Run terminated because of a licensing error.'

        if model_status == '1':
            soln.status = SolutionStatus.optimal
            results.solver.termination_condition = TerminationCondition.optimal
        elif model_status == '2':
            soln.status = SolutionStatus.infeasible
            results.solver.termination_condition = TerminationCondition.infeasible
        elif model_status == '3':
            soln.status = SolutionStatus.unbounded
            results.solver.termination_condition = TerminationCondition.unbounded
        elif model_status == '4':
            soln.status = SolutionStatus.feasible
        elif model_status == '5':
            soln.status = SolutionStatus.unknown

        #
        # Process BARON results file
        #

        # Solutions that were preprocessed infeasible, were aborted, or gave error will not have filled in 
        # res.lst files
        if results.solver.status not in [SolverStatus.error,
                                         SolverStatus.aborted]: 
            #
            # Extract the solution vector and objective value from BARON
            #
            var_value = []
            var_name = []
            var_no = 0
            var_marginal = []
            con_price = []
            SolvedDuringPreprocessing = False          
            
            ##############
            ##Original idea, used a for loop through the file and lots of Flag variables
            #
            #CompFlag = False
            #DualFlag = False
            #SolnVecFlag = False
            #MarginalsFlag = False
            #PriceFlag = False
            #VarNamesFlag = False
            #Skip = 0
            #
            #for line in INPUT:
            #
            #    # The skip variable allows some lines to be ignored after a signal word is 
            #    # found before the relevant lines of information
            #    if Skip > 0:
            #        Skip -= 1
            #        continue
            # 
            #    # If the solution vector was identified:
            #    if SolnVecFlag:
            #        if line != ' \n':
            #            var_no += 1
            #            var_value.append(float(line.split()[2]))
            #        else:
            #            SolnVecFlag = False
            #    # Else if the vector of variable marginals was identified:
            #    elif MarginalsFlag:
            #        if 'Constraint no.' not in line and line != ' \n':
            #            var_marginal.append(float(line.split()[2]))
            #        else:
            #            MarginalsFlag = False
            #    # Else if the vector of constraint prices was identified:
            #    elif PriceFlag:
            #        if line != ' \n':
            #            con_price.append(float(line.split()[2]))
            #        else:
            #            PriceFlag = False
            #    # Else if the vector of variable names was identified:
            #    elif VarNamesFlag:
            #        if line != '\n':
            #            var_name.append(line.split()[0])
            #        else:
            #            VarNamesFlag = False
            #
            #
            #    # This set of conditionals identify the objective, solution vector, and a list
            #    # of the variable names
            #
            #    if CompFlag:
            #        if 'Objective value is:' in line:
            #            objective_value = float(line.split()[4])
            #        elif 'Value\n' in line:
            #            SolnVecFlag = True
            #        elif DualFlag and 'Marginal\n' in line:
            #            MarginalsFlag = True
            #        elif DualFlag and 'Price\n' in line:
            #            PriceFlag = True
            #        elif 'The best solution found is:' in line:
            #            VarNamesFlag = True
            #            Skip = 2
            #        elif 'Corresponding dual solution vector' in line:
            #            DualFlag = True
            #    elif 'Problem solved during preprocessing' in line:
            #        SolvedDuringPreprocessing = True
            #    elif '*** Normal completion ***' in line:
            #        CompFlag = True
            #
            #############


            #############
            #CLH: Simpler code using while loops and .readline().
            #
            # Scan through the first part of the solution file, until the 
            # termination message '*** Normal completion ***'
            line = ''
            while '***' not in line:
                line = INPUT.readline()
                if 'Problem solved during preprocessing' in line:
                    SolvedDuringPreprocessing = True

            INPUT.readline()
            INPUT.readline()
            objective_value = float(INPUT.readline().split()[4])
            INPUT.readline()
            INPUT.readline()

            # Scan through the solution variable values
            line = INPUT.readline()
            while line != ' \n':
                var_value.append(float(line.split()[2]))
                var_no += 1
                line = INPUT.readline()

            # Only scan through the marginal and price values if baron 
            # found that information. 
            if 'Corresponding dual solution vector is' in INPUT.readline():
                has_dual_info = True
                INPUT.readline()
                line = INPUT.readline()
                while 'Price' not in line and line != ' \n':
                    var_marginal.append(float(line.split()[2]))
                    line = INPUT.readline()

                if 'Price' in line:
                    line = INPUT.readline()
                    while line != ' \n':
                        con_price.append(float(line.split()[2]))
                        line = INPUT.readline()

            # Skip either a few blank lines or an empty block of useless
            # marginal and price values (if 'No dual information is available')
            while 'The best solution found is' not in INPUT.readline():
                pass

            # Collect the variable names, which are given in the same order as the lists
            # for values already read
            INPUT.readline()
            INPUT.readline()
            line = INPUT.readline()
            while line != '\n':
                var_name.append(line.split()[0])
                line = INPUT.readline()
            #
            #
            ################

            #
            # Plug gathered information into pyomo soln 
            #
            
            # After collecting solution information, the soln is filled with variable name, number, and value
            # Also, possibly fill the baron_marginal suffix
            for i in xrange(var_no):
                # If using symbolic solver labels, correct variable names to be recognizable to pyomo
                if self.symbolic_solver_labels:
                    var_name[i] = var_name[i].replace('___','(')
                    var_name[i] = var_name[i].replace('__',')')

                var_key = var_name[i]
                soln.variable[var_key]={"Value" : var_value[i], "Id" : i}

                # Only adds the baron_marginal key it is requested and exists
                if extract_marginals and has_dual_info:                  
                    soln.variable[var_key]["baron_marginal"] = var_marginal[i]

            # Fill in the constraint 'price' information
            if extract_price and has_dual_info:
                #CLH: *major* assumption here: that the generator for component_data returns
                #     constraint_data objects in the same order as baron has them listed in
                #     the res.lst file. Baron only provides a number, by the order it read them 
                #     in. This should be the same order that .component_data() uses, since
                #     it was used to write the bar file originally
                i = 0
                for name,index,obj in symbol_map.instance().component_data(ctype=Constraint,active=True):
                    con_label = symbol_map_byObjects[id(obj)]
                    soln.constraint[con_label] = {}
                    soln.constraint[con_label]["baron_price"] = con_price[i]
                    i += 1
            
            # This check is necessary because solutions that are preprocessed infeasible
            # have ok solver status, but no objective value located in the res.lst file
            if not (SolvedDuringPreprocessing and soln.status == SolutionStatus.infeasible): 
                soln.objective[objective_label].value = objective_value
          
            
            ############ Original Idea
            ## Only return a solution when the solver terminates optimally, with unknown status,
            ## or when a time limit is met *and* the solution is feasible.
            #if results.solver.termination_condition in [TerminationCondition.unknown,
            #                                            TerminationCondition.globallyOptimal,
            #                                            TerminationCondition.locallyOptimal,
            #                                            TerminationCondition.optimal,
            #                                            TerminationCondition.other]:
            #    results.solution.insert(soln)
            ##elif (results.solver.termination_condition is TerminationCondition.maxTimeLimit) and (soln.status is not SolutionStatus.infeasible):
            #elif results.solver.termination_condition in [TerminationCondition.maxTimeLimit,
            #                                              TerminationCondition.maxIterations]:
            #    results.solution.insert(soln)
            ##############

            # Better Idea: fill the solution for most cases other than errors
            results.solution.insert(soln)

        INPUT.close()

pyutilib.services.register_executable(name="baron")
