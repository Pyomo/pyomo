#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2010 Sandia Corporation.
#  This software is distributed under the BSD License.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  For more information, see the Pyomo README.txt file.
#  _________________________________________________________________________


import logging
import os

from six import itervalues, iterkeys, iteritems
from six.moves import xrange

logger = logging.getLogger('coopr.solvers')

import pyutilib.services
import pyutilib.common
from pyutilib.misc import Bunch, Options

import pyomo.util._plugin as plugin
from pyomo.opt.base import *
from pyomo.opt.base.solvers import _extract_version
from pyomo.opt.results import *
from pyomo.opt.solver import *
from pyomo.core import SymbolMap, BasicSymbolMap, NumericLabeler, Suffix, TextLabeler
from pyomo.core.numvalue import value
from pyomo.core.block import active_subcomponents_generator, active_subcomponents_data_generator 

from StringIO import StringIO #CLH: I added this to make expr.to_string() work for const. and obj writing
from pyomo.core.objective import Objective 
from pyomo.core import Constraint, Var, Param, Model
from pyomo.core.set_types import * #CLH: added this to be able to recognize variable types when initializing them for baron
from pyomo.core.suffix import active_export_suffix_generator #CLH: EXPORT suffixes "constraint_types" and "branching_priorities" pass their respective information to the .bar file
import re #CLH: added to match the suffixes while processing the solution. Same as CPLEX.py
import tempfile #CLH: added to make a tempfile used to get the version of the baron executable. 
from pyomo.repn.plugins.baron_writer import ProblemWriter_bar

class BARONSHELL(SystemCallSolver):
    """The BARON MINLP solver
    """

    plugin.alias('baron',  doc='The BARON MINLP solver')

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
        
        # The solution file is created in the _convert_problem function.
        # The bar file needs the solution filename in the OPTIONS section, but
        # this function is executed after the bar problem file writing.
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
        
        #print('************************************************')
        #print('Executing _convert_problem in BARON.py plugin')
        #print('************************************************')

        assert len(args) == 1
        instance = args[0]

        problem_filename = pyutilib.services.TempfileManager.create_tempfile(suffix='.pyomo.bar')
        
        self.soln_file = pyutilib.services.TempfileManager.create_tempfile(suffix = '.baron.sol')
        self.tim_file = pyutilib.services.TempfileManager.create_tempfile(suffix = '.baron.tim')

        ###### Handle the writing of OPTIONS before passing control over to the 
        #      baron_writer script
        #
        #
        prob_file = open(problem_filename, 'w')

        #
        # OPTIONS
        #
        
        prob_file.write("OPTIONS{\nResName: \""+self.soln_file+"\";\n")

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

        prob_file.close()
        #
        #
        #######

        #CLH: what should this be? It serves no role in the baron_writer
        solver_capabilities = True

        io_options = {'file_determinism':1, 'symbolic_solver_labels':self.symbolic_solver_labels}

        write_barfile = ProblemWriter_bar()
        output_filename,symbol_map = write_barfile(instance, problem_filename,solver_capabilities,io_options)

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
            if re.match(suffix, "baron_marginal"): #baron_marginal
                extract_marginals = True
                flag = True
            if re.match(suffix, "baron_price"): #baron_price
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
        instance = symbol_map.instance()

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
  
        for block in instance.all_blocks(sort_by_keys=True):
            for name,index,obj in block.subcomponent_data(ctype=Objective,active=True):
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
            
            #############
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
            
            # After collecting solution information, the soln is filled with variable name,
            # number, and value. Also, optionally fill the baron_marginal suffix
            for i in xrange(var_no):

                var_label = var_name[i]

                # If using symbolic solver labels, correct variable names to be recognizable to pyomo
                if self.symbolic_solver_labels:
                    # CLH: The name flipping step is a hack-y way to handle a problem that could
                    # arise if users make variable names with underscores next to indices.
                    # Using str.replace on original name: '_x_(0)' --> '_x____0__' --> '_x(_0)'
                    # Instead, filp so that replace happens from right to left so that variable
                    # names remain intact. 
                    var_label = var_label[::-1]
                    var_label = var_label.replace('___','(')
                    var_label = var_label.replace('__',')')
                    var_label = var_label[::-1]

                var_key = var_label
                soln.variable[var_key]={"Value" : var_value[i], "Id" : i}

                # Only adds the baron_marginal key it is requested and exists
                if extract_marginals and has_dual_info:                  
                    soln.variable[var_key]["baron_marginal"] = var_marginal[i]

            # Fill in the constraint 'price' information
            if extract_price and has_dual_info:
                #CLH: *major* assumption here: that the generator for subcomponent_data returns
                #     constraint_data objects in the same order as baron has them listed in
                #     the res.lst file. Baron only provides a number, by the order it read them 
                #     in. This should be the same order that .subcomponent_data() uses, since
                #     it was used to write the bar file originally
                i = 0
                for block in instance.all_blocks(sort_by_keys=True):
                    for name,index,obj in block.subcomponent_data(ctype=Constraint,active=True):
                        con_label = symbol_map_byObjects[id(obj)]
                        soln.constraint[con_label] = {}
                        soln.constraint[con_label]["baron_price"] = con_price[i]
                        i += 1
            
            # This check is necessary because solutions that are preprocessed infeasible
            # have ok solver status, but no objective value located in the res.lst file
            if not (SolvedDuringPreprocessing and soln.status == SolutionStatus.infeasible): 
                soln.objective[objective_label].value = objective_value
          
            # Fill the solution for most cases, except errors
            results.solution.insert(soln)
            


        INPUT.close()

pyutilib.services.register_executable(name="baron")
