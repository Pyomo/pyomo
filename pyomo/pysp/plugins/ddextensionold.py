# dlw Dec 2014: all 'wb' now 'wt'
#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________
#
# ph extension for dual decomposition
#

import os
import csv
import sys
import itertools

thisfile = os.path.abspath(__file__)


from pyomo.core.base import *
from pyomo.core.base.set_types import *
from pyomo.pysp.plugins.ddextensionnew import (MatrixEntriesClass,
                                               LPFileObjClass,
                                               LPFileConstraintClass)

from six import iteritems

# This must be in your path
ddsip_help = 'DDSIPhelp_x64'

class ddextension_base(object):

    def _getNumberOfStochasticRHS(self, ph):
        raise NotImplementedError

    def _getNumberOfStochasticMatrixEntries(self, ph):
        raise NotImplementedError

    def _sort_print_second_stage_constr(self,
                                        ph,
                                        RememberSecStageConstr,
                                        lp,
                                        ConstraintMap,
                                        count_rows,
                                        MatrixEntries_ConstrToRow_Map):
        raise NotImplementedError

    def _write_rhs_sc(self, ph):
        raise NotImplementedError

    def _write_matrix_sc(self, ph):
        raise NotImplementedError

    def __init__(self):

        self._lpfilename = "ddlpfile.lp"
        self._firststage_var_postfix = '__DDSIP_FIRSTSTAGE'
        self._reference_scenario = None
        self._reference_scenario_instance = None
        self._FirstStageVars = []
        self._FirstStageVarIdMap = {}
        self._SecondStageVars = []
        self._AllVars = []

        self._ScenarioVector = []
        # DG: This is the new column map read from the siphelp output
        #     of column order
        self._ColumnMap = {}

        # keep track of input file names
        self.input_file_name_list = []

        self._precision_string = '.17g'

    def reset(self, ph):
        self.__init__()

    def pre_ph_initialization(self, ph):
        pass

    def post_instance_creation(self, ph):
        pass

    def post_ph_initialization(self, ph):

        print("Hello from the post_ph_initialization callback in ddphextension")

        self._ScenarioVector = \
            sorted(ph._scenario_tree._scenario_map.keys())

        self._reference_scenario = \
            ph._scenario_tree._scenario_map[self._ScenarioVector[0]]
        scenario_name = self._reference_scenario._name

        print(("\nUsing %s as reference scenario" % (scenario_name)))

        if isinstance(ph._solver_manager,
                      pyomo.solvers.plugins.smanager.phpyro.SolverManager_PHPyro):
            # If this is parallel ph, the instances do not exist on
            # this process, so let's construct the one we need
            singleton_tree = ph._scenario_tree._scenario_instance_factory.generate_scenario_tree()
            singleton_tree.compress([scenario_name])
            singleton_dict = singleton_tree._scenario_instance_factory.\
                             construct_instances_for_scenario_tree(
                                 singleton_tree,
                                 output_instance_construction_time=ph._output_instance_construction_time)
            # with the scenario instances now available, link the
            # referenced objects directly into the scenario tree.
            singleton_tree.linkInInstances(singleton_dict,
                                           create_variable_ids=True)

            self._reference_scenario = singleton_tree._scenarios[0]
            scenario_instance = self._reference_scenario._instance
            create_block_symbol_maps(scenario_instance, (Var,))

        else:
            scenario_instance = self._reference_scenario._instance
        self._reference_scenario_instance = scenario_instance

        print("Creating the sip.in file")

        # This is usuall called just prior to solving the instances,
        # however the scenariotree uses CanonicalRepn for determining
        # constraint stage
        ph._preprocess_scenario_instances()

        # Write the lp file for an arbitrary scenario, also obtain the
        # maximum row/column symbol length (in characters)
        max_name_len, symbol_map = self._write_reference_scenario_lp(ph)

        # variables will stored by their name in the LP file,
        # which may be different from what is stored on the
        # ph scenario tree
        StageToConstraintMap = self._Constraints_Stages(ph, symbol_map)
        self._Populate_StageVars(ph, symbol_map)

        # DG: some system calls to call the ddsip help utility to get
        #     column orders Current working directory is ddsip_FARMER2
        #     --- not sure what changes the pwd to this directory
        assert os.path.exists(self._lpfilename)
        ddsip_help_output = "rows+cols"
        os.system("rm -f "+ddsip_help_output)
        os.system("rm -f "+ddsip_help_output+".gz")
        print(("COMMAND= "+str(ddsip_help)+' '+self._lpfilename+' '+str(max_name_len)))
        os.system(ddsip_help+' '+self._lpfilename+' '+str(max_name_len))
        assert os.path.exists(ddsip_help_output+".gz")
        os.system("gzip -df "+ddsip_help_output+".gz")

        # DG: Now parse the rows+cols file:
        print("Opening the rows+cols file for reading")
        dashcount = 0
        varcount = 0
        with open(ddsip_help_output,'r') as f:
            for line in f.readlines():
                pieces = line.split()
                if dashcount == 2:
                    break
                if pieces[0] =='__________________________':
                    dashcount = dashcount+1
                if len(pieces) == 2:
                    varcount = varcount + 1
                    self._ColumnMap[pieces[1]] = int(pieces[0])

        ObjObject, ConstraintMap, remaining_lpfile_rows, AllConstraintNames = \
            self._read_parse_lp(self._lpfilename)

        MatrixEntries_ConstrToRow_Map = \
            self.sort_write_lp(ph,
                               ObjObject,
                               ConstraintMap,
                               StageToConstraintMap,
                               remaining_lpfile_rows)
        self._write_rhs_sc(ph)
        self._write_matrix_sc(ph,
                              MatrixEntries_ConstrToRow_Map)

        self.write_sip_in(ph,
                          StageToConstraintMap,
                          AllConstraintNames)

        self.write_input2sip(ph)
        self.write_input2sip(ph)

    def post_iteration_0_solves(self, ph):
        pass

    def post_iteration_0(self, ph):
        pass

    def pre_iteration_k_solves(self, ph):
        pass

    def post_iteration_k_solves(self, ph):
        pass

    def post_iteration_k(self, ph):
        pass

    def post_ph_execution(self, ph):
        self.write_start_weights(ph)
        ##self.write_start_in(ph)

    # Write the lp file for a scenario and return the maximum
    # character count for names in the file
    def _write_reference_scenario_lp(self, ph):

        # Make sure the pyomo plugins are loaded
        import pyomo.environ
        lp_file_writer = pyomo.repn.plugins.cpxlp.ProblemWriter_cpxlp()

        # Write the LP file
        print(("Writing LP file to %s" % (self._lpfilename,)))
        scenariotree_vars = \
            self._reference_scenario_instance.\
                _ScenarioTreeSymbolMap.bySymbol
        rootnode_vars = \
            ph._scenario_tree.findRootNode()._standard_variable_ids
        firststage_ids = \
            [id(vardata) for scenariotree_id, vardata \
             in iteritems(scenariotree_vars) \
             if scenariotree_id in rootnode_vars]
        capabilities = lambda x: True
        text_labeler = TextLabeler()
        labeler = lambda x: text_labeler(x) + \
                  (self._firststage_var_postfix \
                   if (id(x) in firststage_ids) else \
                   "")
        output_filename, symbol_map = \
            lp_file_writer(self._reference_scenario_instance,
                           self._lpfilename,
                           capabilities,
                           {'labeler' : labeler})
        assert output_filename == self._lpfilename
        print("\nModel written\n")

        # Return the max symbol length (for siphelp)
        max_name_len = max(len(symbol) for symbol in symbol_map.bySymbol)
        return max_name_len, symbol_map

    def _Populate_StageVars(self, ph, LP_symbol_map):

        all_vars_cnt = 0
        for block in self._reference_scenario_instance.block_data_objects(active=True):
            all_vars_cnt += len(list(components_data(block, Var)))

        rootnode = ph._scenario_tree.findRootNode()
        stagetwo = ph._scenario_tree._stages[1]
        anode = stagetwo._tree_nodes[0]
        firststage_blended_variables = rootnode._standard_variable_ids
        LP_byObject = LP_symbol_map.byObject
        for scenario_tree_id, vardata in \
              iteritems(self._reference_scenario_instance.\
              _ScenarioTreeSymbolMap.bySymbol):
            try:
                LP_name = LP_byObject[id(vardata)]
            except:
                print(("FAILED ON VAR DATA= "+vardata.name))
                foobar
            if scenario_tree_id in firststage_blended_variables:
                self._FirstStageVars.append(LP_name)
                self._FirstStageVarIdMap[LP_name] = scenario_tree_id
            elif (scenario_tree_id in rootnode._derived_variable_ids) or \
                 (scenario_tree_id in anode._variable_ids):
                self._SecondStageVars.append(LP_name)
            else:
                # More than two stages?
                assert False
            self._AllVars.append(LP_name)

        for stage in ph._scenario_tree._stages:
            cost_variable_name, cost_variable_index = \
                stage._cost_variable
            stage_cost_component = \
                self._reference_scenario_instance.\
                find_component(cost_variable_name)
            if stage_cost_component.type() is not Expression:
                LP_name = LP_byObject[id(stage_cost_component[cost_variable_index])]
                assert LP_name not in self._FirstStageVars
                if LP_name not in self._AllVars:
                    assert LP_name not in self._SecondStageVars
                    self._SecondStageVars.append(LP_name)
                    self._AllVars.append(LP_name)

        # Make sure every variable on the model has been
        # declared on the scenario tree
        if len(self._AllVars) != all_vars_cnt:
            print("**** THERE IS A PROBLEM ****")
            print("Not all model variables are on the scenario tree. Investigating...")
            all_vars = set()
            for block in self._reference_scenario_instance.block_data_objects(active=True):
                all_vars.update(vardata.name \
                                for vardata in components_data(block, Var))
            tree_vars = set()
            for scenario_tree_id, vardata in \
                iteritems(self._reference_scenario_instance.\
                          _ScenarioTreeSymbolMap.bySymbol):
                tree_vars.add(vardata.name)
            cost_vars = set()
            for stage in ph._scenario_tree._stages:
                cost_variable_name, cost_variable_index = \
                    stage._cost_variable
                stage_cost_component = \
                    self._reference_scenario_instance.\
                    find_component(cost_variable_name)
                if stage_cost_component.type() is not Expression:
                    cost_vars.add(stage_cost_component[cost_variable_index].name)
            print(("Number of Scenario Tree Variables (found ddsip LP file): "+str(len(tree_vars))))
            print(("Number of Scenario Tree Cost Variables (found ddsip LP file): "+str(len(cost_vars))))
            print(("Number of Variables Found on Model: "+str(len(all_vars))))
            print(("Variables Missing from Scenario Tree (or LP file):"+str(all_vars-tree_vars-cost_vars)))


        # A necessary but not sufficient sanity check to make sure the
        # second stage variable sets are the same for all
        # scenarios. This is not required by pysp, but I think this
        # assumption is made in the rest of the code here
        for tree_node in stagetwo._tree_nodes:
            assert len(anode._variable_ids) == \
                len(tree_node._variable_ids)

        assert len(ph._scenario_tree._stages) == 2

        # we are doing string-sort
        self._FirstStageVars.sort()
        self._SecondStageVars.sort()
        self._AllVars.sort()

    def _Constraints_Stages(self, ph, LP_symbol_map):

        # save the output in a map: key is constraint name, value is a
        # constraintindex-stage-map
        #ConstraintToStageMap = {}
        # inverse map to ConstraintToStageMap: key is stage name,
        # value is constraintname (which includes the index)
        StageToConstraintMap = {}
        # auxiliary list for StageToConstraintMap
        FirstStageConstrNameToIndex = []
        # auxiliary list for StageToConstraintMap
        SecondStageConstrNameToIndex = []

        stage1 = ph._scenario_tree._stages[0]
        stage2 = ph._scenario_tree._stages[1]

        reference_instance = self._reference_scenario_instance
        LP_byObject = LP_symbol_map.byObject
        # deal with the fact that the LP writer prepends constraint
        # names with things like 'c_e_', 'c_l_', etc depending on the
        # constraint bound type and will even split a constraint into
        # two constraints if it has two bounds

        reference_scenario = self._reference_scenario

        LP_reverse_alias = dict()
        for symbol in LP_symbol_map.bySymbol:
            LP_reverse_alias[symbol] = []
        for alias, obj_weakref in iteritems(LP_symbol_map.aliases):
            LP_reverse_alias[LP_byObject[id(obj_weakref())]].append(alias)
        for block in reference_instance.block_data_objects(active=True):
            block_canonical_repn = getattr(block, "_canonical_repn",None)
            if block_canonical_repn is None:
                raise ValueError("Unable to find _canonical_repn ComponentMap "
                                 "on block %s" % (block.name))
            isPiecewise = False
            if isinstance(block, (Piecewise, _PiecewiseData)):
                isPiecewise = True
            for constraint_data in block.component_data_objects(SOSConstraint, active=True, descend_into=False):
                raise TypeError("SOSConstraints are not handled by the DDSIP interface: %s"
                                % (constraint_data.name))
            for constraint_data in block.component_data_objects(Constraint, active=True, descend_into=False):
                LP_name = LP_byObject[id(constraint_data)]
                # if it is a range constraint this will account for
                # that fact and hold and alias for each bound
                LP_aliases = LP_reverse_alias[LP_name]
                assert len(LP_aliases) > 0
                if not isPiecewise:
                    constraint_node = reference_scenario.constraintNode(
                        constraint_data,
                        canonical_repn=block_canonical_repn.get(constraint_data),
                        instance=reference_instance)
                    stage_index = reference_scenario.node_stage_index(constraint_node)
                else:
                    stage_index = 1
                stage_index = reference_scenario.node_stage_index(constraint_node)
                if stage_index == 0:
                    FirstStageConstrNameToIndex.extend(LP_aliases)
                elif stage_index == 1:
                    SecondStageConstrNameToIndex.extend(LP_aliases)
                else:
                    # More than two stages?
                    assert False

        StageToConstraintMap['FirstStage'] = FirstStageConstrNameToIndex
        StageToConstraintMap['SecondStage'] = SecondStageConstrNameToIndex

        return StageToConstraintMap

    def write_sip_in(self, ph, StageToConstraintMap, AllConstraintNames):
        try:
            print("\n\nWrite dd input file: sip.in \n")
            sipin = open("sip.in", "w")
        except IOError:
            print("sip.in cannot be created!")
            sys.exit(1)

        # keep track of input file names
        self.input_file_name_list.append('sip.in')

        FirstStage = StageToConstraintMap['FirstStage']
        NumberOfFirstStageConstraints = 0
        for constr in FirstStage:
            assert constr in AllConstraintNames
            NumberOfFirstStageConstraints += 1

        SecondStage = StageToConstraintMap['SecondStage']
        NumberOfSecondStageConstraints = 0
        for constr in SecondStage:
            assert constr in AllConstraintNames
            NumberOfSecondStageConstraints += 1

        NumberOfFirstStageVars = len(self._FirstStageVars)
        NumberOfSecondStageVars = len(self._SecondStageVars)

        NumberOfStochasticRHS = self._getNumberOfStochasticRHS(ph)
        NumberOfStochasticMatrixEntries = self._getNumberOfStochasticMatrixEntries(ph)

        NumberOfScenarios = len(self._ScenarioVector)

        sipin.write('BEGIN \n\n\n')
        sipin.write('FIRSTCON '+str(NumberOfFirstStageConstraints)+'\n')
        sipin.write('FIRSTVAR '+str(NumberOfFirstStageVars)+'\n')
        # NOTE: The "+1" terms below are due to the ONE_VAR_CONSTANT
        #       variable definition
        sipin.write('SECCON '+str(NumberOfSecondStageConstraints+1)+'\n')
        sipin.write('SECVAR '+str(NumberOfSecondStageVars+1)+'\n')
        sipin.write('POSTFIX '+self._firststage_var_postfix+'\n')
        sipin.write('SCENAR '+str(NumberOfScenarios)+'\n')

        sipin.write('STOCRHS '+str(NumberOfStochasticRHS)+'\n')
        sipin.write('STOCCOST '+str('0')+'\n')
        sipin.write('STOCMAT '+str(NumberOfStochasticMatrixEntries)+'\n')

        sipin.write("\n\nCPLEXBEGIN\n")
        sipin.write('1035 0 * Output on screen indicator\n')
        sipin.write('2008 0.001 * Absolute Gap\n')
        sipin.write('2009 0.001 * Relative Gap\n')
        sipin.write('1039 1200 * Time limit\n')
        sipin.write('1016 1e-9 * simplex feasibility tolerance\n')
        sipin.write('1014 1e-9 * simplex optimality tolerance\n')
        sipin.write('1065 40000 * Memory available for working storage\n')
        sipin.write('2010 1e-20 * integrality tolerance\n')
        sipin.write('2008 0 * Absolute gap\n')
        sipin.write('2020 0 * Priority order\n')
        sipin.write('2012 4 * MIP display level\n')
        sipin.write('2053 2 * disjunctive cuts\n')
        sipin.write('2040 2 * flow cover cuts\n')
        sipin.write('2060 3 *DiveType mip strategy dive (probe=3)\n')
        sipin.write('CPLEXEND\n\n')

        sipin.write('MAXINHERIT 15\n')
        sipin.write('OUTLEV 5 * Debugging\n')
        sipin.write('OUTFIL 2\n')
        sipin.write('STARTI 0  * (1 to use the starting values from PH)\n')
        sipin.write('NODELI 2000 * Sipdual node limit\n')
        sipin.write('TIMELIMIT 964000 * Sipdual time limit\n')
        sipin.write('HEURISTIC 99 3 7 * Heuristics: Down, Up, Near, Common, Byaverage ...(12)\n')
        sipin.write('ABSOLUTEGAP 0.001 * Absolute duality gap allowed in DD\n')
        sipin.write('EEVPROB 1\n')
        sipin.write('RELATIVEGAP 0.01 * (0.02) Relative duality gap allowed in DD\n')
        sipin.write('BRADIRECTION -1 * Branching direction in DD\n')
        sipin.write('BRASTRATEGY 1 * Branching strategy in DD (1 = unsolved nodes first, 0 = best bound)\n')
        sipin.write('EPSILON 1e-13 * Branch epsilon for cont. var.\n')
        sipin.write('ACCURACY 5e-16 * Accuracy\n')
        sipin.write('BOUSTRATEGY 1 * Bounding strategy in DD\n')
        sipin.write('NULLDISP 1e-16\n')
        sipin.write('RELAXF 0\n')
        sipin.write('INTFIRST 0 * Branch first on integer\n')
        sipin.write('HOTSTART 4 * use previous solution as integer starting info\n')

        sipin.write('\n\nRISKMO 0 * Risk Model\n')
        sipin.write('RISKALG 1\n')
        sipin.write('WEIGHT 1\n')
        sipin.write('TARGET 54 * target if needed\n')
        sipin.write('PROBLEV .8 * probability level\n')
        sipin.write('RISKBM 11000000 * big M in \n')

        sipin.write('\n\nCBFREQ 0 * (50) Conic Bundle in every ith node\n')
        sipin.write('CBITLIM 20 * (10) Descent iteration limit for conic bundle method\n')
        sipin.write('CBTOTITLIM 50 * (1000) Total iteration limit for conic bundle method\n')
        sipin.write('NONANT 1 * Non-anticipativity representation\n')
        sipin.write('DETEQU 1 * Write Deterministic Equivalent\n')

        sipin.write('\n\nEND\n')

        sipin.close()

    def _read_unique_coef(self):
        # read the files containing stochastic coefficients which are
        # given by the user
        try:
            MAT_datafile = csv.reader(open("StochDataMAT.dat", "rb"), delimiter=' ', quotechar='|', skipinitialspace=True)
            RHS_datafile = csv.reader(open("StochDataRHS.dat", "rb"), delimiter=' ', quotechar='|', skipinitialspace=True)
        except csv.Error as e:
            print("StochData is not found!")
            sys.exit('file %s, line %d: %s' % (filename, f.line_num, e))

        # key is constraint name, value is a class with a map od
        # variable name and unique value
        stochMAT = {}
        # key is constraintname, value is stochastic rhs
        stochRHS = {}

        for line in MAT_datafile:
            name = line[1]
            keys = list(stochMAT.keys())
            if name not in keys:
                stochMAT[name] = MatrixEntriesClass()
            stochMAT[name].AddToMap(line)

        for line in RHS_datafile:
            constr_name = line[1]
            value = line[2]
            stochRHS[constr_name] = value

        return stochMAT, stochRHS

    def _read_parse_lp(self, lp_filename):

        # read and parse the lp file which was generated by
        # lp_file_writer above in post_ph_initialization(self, ph)

        # This class has the OBJ sense, the OBJ name and a
        # variable-value-dictionary
        ObjObject = LPFileObjClass()

        # The keys are the constraint names, value is a class with a
        # variable-value-dictionary, the comparator and the RHS.
        ConstraintMap = {}

        try:
            print(("\nName of the lp file which will be read: "+str(lp_filename)+"\n"))
#dlw Dec2014            lp_file = csv.reader(open(lp_filename, 'rb'), delimiter=' ', quotechar='|', skipinitialspace=True)
            lp_file = csv.reader(open(lp_filename, 'rt'), delimiter=' ', quotechar='|', skipinitialspace=True)
        except csv.Error as e:
            print((lp_filename+" is not found!"))
            sys.exit('file %s, line %d: %s' % (filename, f.line_num, e))

        # increment this only after you have processed the row
        list_lp_file = list(lp_file)[0:]

        CurrRow = 0
        CurrRow = self.SkipLPNullLines(CurrRow, list_lp_file)

        # assume it is a sense
        ObjObject.AssignSense(list_lp_file[CurrRow][0])
        CurrRow += 1

        CurrElem = 0
        # now assume we are on to the name
        name = self.ClearName(list_lp_file[CurrRow][CurrElem])
        ObjObject.AssignName(name)
        CurrRow, CurrElem = \
            self.LocalPostionUpdate(list_lp_file, CurrRow, CurrElem, 1)

        # send the pairs to the obj object
        # assume the obj is followed by a blank line!!!
        while list_lp_file[CurrRow] != []:
            LastCoeff = None
            # process the pairs
            if LastCoeff is not None:
                ObjObject.AddToMap(LastCoeff, list_lp_file[CurrRow][CurrElem])
                LastCoeff = None
                CurrRow, CurrElem = self.LocalPostionUpdate(list_lp_file, CurrRow, CurrElem, 1)
            elif 1+CurrElem == len(list_lp_file[CurrRow]):
                LastCoeff = list_lp_file[CurrRow][CurrElem]
                CurrRow, CurrElem = self.LocalPostionUpdate(list_lp_file, CurrRow, CurrElem, 1)
            else:
                ObjObject.AddToMap(list_lp_file[CurrRow][CurrElem], list_lp_file[CurrRow][CurrElem+1])
                LastCoeff = None
                CurrRow, CurrElem = self.LocalPostionUpdate(list_lp_file, CurrRow, CurrElem, 2)

        if LastCoeff is not None:
            print("\nerror mismatched pairs in objective function")
            exit(1)

        #print("ObjObject = "+str(ObjObject()))

        CurrRow = self.SkipLPNullLines(CurrRow, list_lp_file)
        CurrElem = 0
        # check if we reached the constraints
        if (('s.t.' in list_lp_file[CurrRow][CurrElem]) or \
            ('st' in list_lp_file[CurrRow][CurrElem]) or \
            ('subject to' in list_lp_file[CurrRow][CurrElem])):

            CurrRow += 1
            CurrRow = self.SkipLPNullLines(CurrRow, list_lp_file)

            while (('bound' not in list_lp_file[CurrRow][0]) and \
                   ('c_e_ONE_VAR_CONSTANT' not in list_lp_file[CurrRow][0])):

                CurrConstraintName = self.ClearName(list_lp_file[CurrRow][CurrElem])
                ConstraintMap[CurrConstraintName] = LPFileConstraintClass()
                if len(list_lp_file[CurrRow]) > 1:
                    CurrElem += 1
                else:
                    CurrRow += 1

                LastCoeff = None
                end = ''
                while list_lp_file[CurrRow] != []:
                    # process the pairs, keep track of last coefficient and
                    # variable, just in case we reach the end of the
                    # constraint (AssignComparator and AssignRHS need
                    # them)
                    if LastCoeff is not None:
                        end = ConstraintMap[CurrConstraintName].AddToMap(LastCoeff, list_lp_file[CurrRow][CurrElem])
                        LastVar = list_lp_file[CurrRow][CurrElem]
                        #print "in 1. IF: LastCoeff, LastVar =", LastCoeff, LastVar; sys.stdout.flush()
                        if end != 'EndOfConstraint':
                            LastCoeff = None
                            CurrRow, CurrElem = self.LocalPostionUpdate(list_lp_file, CurrRow, CurrElem, 1)
                    elif 1+CurrElem == len(list_lp_file[CurrRow]):
                        LastCoeff = list_lp_file[CurrRow][CurrElem]
                        CurrRow, CurrElem = self.LocalPostionUpdate(list_lp_file, CurrRow, CurrElem, 1)
                        #print "in 1. ElIF: LastCoeff=", LastCoeff; sys.stdout.flush()
                    elif 2+CurrElem <= len(list_lp_file[CurrRow]):
                        end = ConstraintMap[CurrConstraintName].AddToMap(list_lp_file[CurrRow][CurrElem], list_lp_file[CurrRow][CurrElem+1])
                        LastCoeff = list_lp_file[CurrRow][CurrElem]
                        LastVar = list_lp_file[CurrRow][CurrElem+1]
                        #print "in 2. ElIF: LastCoeff, LastVar =", LastCoeff, LastVar; sys.stdout.flush()
                        if end != 'EndOfConstraint':
                            LastCoeff = None
                            CurrRow, CurrElem = self.LocalPostionUpdate(list_lp_file, CurrRow, CurrElem, 2)
                    if end == 'EndOfConstraint':
                        #print "\nend of constraint"
                        ConstraintMap[CurrConstraintName].AssignComparator(LastCoeff)
                        ConstraintMap[CurrConstraintName].AssignRHS(LastVar)
                        #print "in 2. IF: LastCoeff, LastVar =", LastCoeff, LastVar; sys.stdout.flush()
                        CurrRow += 2
                        CurrElem = 0
                        break

        remaining_lpfile_rows = []
        while CurrRow < len(list_lp_file):
            remaining_lpfile_rows.append(list_lp_file[CurrRow])
            CurrRow += 1

        AllConstraintNames = list(ConstraintMap.keys())

        return ObjObject, ConstraintMap, remaining_lpfile_rows, AllConstraintNames

    def sort_write_lp(self, ph, ObjObject, ConstraintMap, StageToConstraintMap, remaining_lpfile_rows):

        try:
            print("\nWrite the LP file for dd in sorted_LPfile.lp\n")
            lp = open("sorted_LPfile.lp", "w")
        except IOError:
            print("IO Error so that sorted_LPfile.lp cannot be created.")
            sys.out(1)

        # keep track of input file names
        self.input_file_name_list.append('sorted_LPfile.lp')

        # for the matrix.sc file we need to know which constraint is
        # in which row:
        MatrixEntries_ConstrToRow_Map = {}

        lp.write(ObjObject[0]+"\n "+ObjObject[1]+":\n")
        self.print_coeff_var_from_map(ObjObject[2], lp)

        # assume blank line before and after constraint indicator
        lp.write("\ns.t.\n")

        FirstStage = StageToConstraintMap['FirstStage']
        ConstrNames = list(ConstraintMap.keys())
        ConstrNames.sort()
        RememberSecStageConstr = []
        # so that we know in which rows the constraints with
        # stochastic data are (first row has index 0)
        count_rows = -1

        for name in ConstrNames:
            # check if constraint is in first stage
            if name in FirstStage:
                lp.write("\n"+name+":\n")
                count_rows += 1
                MatrixEntries_ConstrToRow_Map[name] = count_rows
                self.print_coeff_var_from_map(ConstraintMap[name][0], lp)
                lp.write(ConstraintMap[name][1]+" "+ConstraintMap[name][2]+"\n")
            else:
                RememberSecStageConstr.append(name)

        MatrixEntries_ConstrToRow_Map = \
            self._sort_print_second_stage_constr(ph,
                                                 RememberSecStageConstr,
                                                 lp,
                                                 ConstraintMap,
                                                 count_rows,
                                                 MatrixEntries_ConstrToRow_Map)

        # print the remaining rows of the lp file
        for row in range(len(remaining_lpfile_rows)):
            lp.write("\n")
            for i in range(len(remaining_lpfile_rows[row])):
                lp.write(remaining_lpfile_rows[row][i]+" ")

        lp.close()
        #print MatrixEntries_ConstrToRow_Map
        return MatrixEntries_ConstrToRow_Map

    def write_start_weights(self, ph):

        rootnode_name = ph._scenario_tree.findRootNode()._name
        weights_vectors = []
        for scenarioname in self._ScenarioVector:
            scenario_weights = \
                ph._scenario_tree._scenario_map[scenarioname]._w[rootnode_name]
            scenario_weights_vector = []
            for varname_LP in self._FirstStageVars:
                scenario_tree_id = self._FirstStageVarIdMap[varname_LP]
                scenario_weights_vector.append(scenario_weights[scenario_tree_id])
            weights_vectors.append(scenario_weights_vector)

        # transpose to orient data for processing across scenarios
        weights_vectors = list(zip(*weights_vectors))

        with open('PHWEIGHTS.csv','wt') as f:
            f.write("varname,"+(",".join(name for name in self._ScenarioVector))+"\n")
            for var_index, vector_w in enumerate(weights_vectors):
                varname = self._FirstStageVars[var_index]
                f.write(varname+","+(",".join(repr(w) for w in vector_w))+"\n")

        num_scenarios = float(len(self._ScenarioVector))

        # NONANT 1 case:
        v1_transpose = []
        for vector_w in weights_vectors:
            # all but the first position
            vector_w_iter = vector_w[1:]
            tmp = [-w/num_scenarios \
                   for w in vector_w_iter]
            v1_transpose.append(tmp)
        with open('NONANT1.in','wt') as f:
            f.write('MULTIPLIER\n')
            # transpose and flatten
            for column in zip(*v1_transpose):
                f.writelines(repr(x)+'\n' for x in column)

        # NONANT 2 case:
        v2_transpose = []
        for vector_w in weights_vectors:
            w_sum = 0.0
            tmp = []
            # all but the last element
            vector_w_iter = vector_w[:-1]
            for w in vector_w_iter:
                w_sum += w
                tmp.append(w_sum/num_scenarios)
            v2_transpose.append(tmp)
        with open('NONANT2.in','wt') as f:
            f.write('MULTIPLIER\n')
            # tranpose and flatten
            for column in zip(*v2_transpose):
                f.writelines(repr(x)+'\n' for x in column)

        # NONANT 3 case:
        v3_transpose = []
        for vector_w in weights_vectors:
            wlast = vector_w[-1]
            # all but the last element
            vector_w_iter = vector_w[:-1]
            tmp = [(w-wlast)/num_scenarios for w in vector_w_iter]
            v3_transpose.append(tmp)
        with open('NONANT3.in','wt') as f:
            f.write('MULTIPLIER\n')
            # tranpose and flatten
            for column in zip(*v3_transpose):
                f.writelines(repr(x)+'\n' for x in column)

    def write_start_in(self, ph):
        ### Note: as of Dec 26, 2014, the ph object we have here does not have the
        ###       scenario tree that has the solution, so this will not work
        ###       I put this code in a solution writer to get what I need - DLW
        # keep track of input file names -- or not (dlw Dec 2014)
        ### self.input_file_name_list.append('start.in')

        print("\n\nWrite dd input file: solstart.in (** integers will be rounded ** )\n")

        # note: DDSIP seems to get the names in alpha order from cplex
        VNames = []
        VVals = {}
        # assume two-stage and we only want the first stage vars
        rootnode = ph._scenario_tree.findRootNode()
        for variable_id in rootnode._variable_ids:
            var_name, index = rootnode._variable_ids[variable_id]
            name = str(var_name)+str(index)
            name = name.replace('.','_') # lp files are what matters here
            name = name.replace(',',':')
            name = name.replace(' ','')
            VNames.append(name)
            ### VVals[name] = rootnode._solution[variable_id]
            VVals[name] = rootnode.get_variable_value(var_name, index)
        VNames.sort()
        print ("Writing init solution:")
        with open('solstart.in', 'wt') as f: 
           f.write("SOLUTION\n")
           for name in VNames:
                if rootnode.is_variable_discrete(variable_id):
                    val = str(round(float(VVals[name])))
                else:
                    val = str(float(VVals[name]))
                f.write(val+"\n") 
                # aside: rounding changes slightly from Python 2.x to 3.x

    def write_input2sip(self, ph):
        try:
            print("\n\nWrite dd input file: input2sip \n")
            f = open('2sip', 'w')
        except IOError:
            print("File input2sip cannot be created.")
            sys.exit(1)

        if 'sip.in' in self.input_file_name_list:
            f.write("sip.in\n")
        if 'sorted_LPfile.lp' in self.input_file_name_list:
            f.write("sorted_LPfile.lp\n")
        if 'model.ord' in self.input_file_name_list:
            f.write("model.ord\n")
        if 'rhs.sc' in self.input_file_name_list:
            f.write("rhs.sc\n")
        if 'cost.sc' in self.input_file_name_list:
            f.write("cost.sc\n")
        if 'matrix.sc' in self.input_file_name_list:
            f.write("matrix.sc\n")
        if 'order.dat' in self.input_file_name_list:
            f.write("order.dat\n")
        if 'start.in' in self.input_file_name_list:
            f.write("start.in\n")

        f.close()

    def SkipLPNullLines(self, CurrRow, lp_file):
        # skip over comment lines and blank lines (assuming comment
        # character leads)
        while (lp_file[CurrRow] == [] or lp_file[CurrRow][0].startswith("\\")):
            CurrRow += 1
            if CurrRow >= len(lp_file):
                print("Error: past end of file while skipping null lines")
                sys.exit(1)
        return CurrRow

    def LocalPostionUpdate(self, lp_file, CurrRow, CurrElem, step):
        if ((step !=1) and (step != 2)):
            print(("unexpected step length  ="+str(step)))
        CurrElem += step
        if CurrElem > len(lp_file[CurrRow])-1:
            CurrRow += 1
            CurrElem = 0
        return CurrRow, CurrElem

    def ClearName(self, name):
        if name[-1] == ':':
            name = name[:-1]
        return name

    def print_coeff_var_from_map(self, VarToCoeffMap, lp_file):
        keys = list(VarToCoeffMap.keys())
        #string sorting
        keys.sort()
        line_template = "%+"+self._precision_string+" %s\n"
        for key in keys:
            lp_file.write(line_template % (VarToCoeffMap[key], key))
            """
            if ('+' not in str(VarToCoeffMap[key])) and ('-' not in str(VarToCoeffMap[key])):
                lp_file.write("+"+str(VarToCoeffMap[key])+" "+str(key)+"\n")
            else:
                lp_file.write(str(VarToCoeffMap[key])+" "+str(key)+"\n")
            """
    def print_coeff_var(self, coeff, var, lp_file):
        if coeff not in ('=', '<', '>', '<=', '>='):
            if ('+' not in str(coeff)) and ('-' not in str(coeff)):
                lp_file.write("+"+str(coeff)+" "+str(var)+"\n")
            else:
                lp_file.write(str(coeff)+" "+str(var)+"\n")
        else:
            lp_file.write(str(coeff)+" "+str(var)+"\n")
