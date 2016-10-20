# dlw Dec 2014: all 'wb' now 'wt'
# dlw Jan 2015: even more 'w' and 'wb' now 'wt'
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

import csv
import sys
import itertools
from operator import itemgetter
import os
thisfile = os.path.abspath(__file__)

import pyomo.util.plugin
from pyomo.core import *
from pyomo.core.base.var import _VarData
from pyomo.core.base.piecewise import _PiecewiseData
from pyomo.pysp import phextension
from pyomo.pysp.phsolverserver import _PHSolverServer
from pyomo.pysp.phsolverserverutils import \
    transmit_external_function_invocation_to_worker

from six import iteritems

# This must be in your path
_ddsip_help = 'DDSIPhelp_x64'
_ddsip_help_output = "rows+cols"
_ddsip_exe = 'DDSIP_126_x64'
_precision_string = '.17g'

_disable_stoch_rhs_flagname = "DDSIP_DISABLE_STOCHASTIC_RHS"
_disable_stoch_rhs_default = False

_disable_stoch_matrix_flagname = "DDSIP_DISABLE_STOCHASTIC_MATRIX"
_disable_stoch_matrix_default = True

_disable_stoch_costs_flagname = "DDSIP_DISABLE_STOCHASTIC_COSTS"
_disable_stoch_costs_default = True

# TODO:
# handle the case where a variable on the scenario tree does not
#  appear in the LP file
# test with range constraints
# test with == constraints
# test with <= constraints
# test with >= constraints
# test with Piecewise
# test that SOSConstraint throws error
# test number of stochastic entries match
#  (this is not a necessary or sufficient condition
#   for checking that something has gone wrong, but
#   it should catch most issues)

# POTENTIAL ISSUES
# The fact that the constraint matrix is sparse presents a problem when
# stochastic matrix coefficients are sometimes zero. In these cases, somewhere
# in the pyomo toolchain these entries will be simplified out of the constraint
# matrix (e.g, when generating the pyomo expression or when writing the
# LP file lines). In such a case, the generated stochastic matrix file might be
# garbage (but not cause an error).
#
# In a related case, if a constraint becomes "trivial" (due to stochastic coefficients)
# that could cause it appear in some scenario models but not others. Again, this
# could possibly result in garbage DDSIP inputs that don't cause an error.
#
# My proposed solution is choose a global reference scenario that all other scenarios
# must compare nonzero locations against as they write their own file. This is not
# implemented yet.

def Write_DDSIP_Input(ph, scenario_tree, scenario_or_bundle,
                      scenario_name,
                      scenario_index,
                      firststage_var_suffix,
                      global_reference_scenario=False):
    dd = DDSIP_Input(ph, scenario_tree, scenario_or_bundle,
                     scenario_name,
                     scenario_index,
                     firststage_var_suffix)
    return dd.write(ph, global_reference_scenario=global_reference_scenario)

class DDSIP_Input(object):

    def __init__(self,
                 ph, scenario_tree, scenario_or_bundle,
                 scenario_name,
                 scenario_index,
                 firststage_var_suffix):

        self.input_file_name_list = []
        self._scenario_name = scenario_name
        self._scenario_index = scenario_index
        self._lpfilename = "ddlpfile.lp."+self._scenario_name
        self._rhsfilename = "rhs.sc."+self._scenario_name
        self._matfilename = "matrix.sc."+self._scenario_name
        self._objfilename = "cost.sc."+self._scenario_name
        self._firststage_var_suffix = firststage_var_suffix
        self._reference_scenario = ph._scenario_tree.get_scenario(scenario_name)
        self._reference_scenario_instance = None

        self._FirstStageVars = []
        self._FirstStageVarIdMap = {}
        self._FirstStageDerivedVars = []
        self._FirstStageDerivedVarIdMap = {}
        self._SecondStageVars = []
        self._SecondStageVarIdMap = {}
        self._StageCostVarStageMap = {}

        # If these values remain as None, it will
        # indicate the user has explicitely set
        # disable flags on the model (which
        # we will report for confirmation purposes)
        self._num_stochastic_rhs = None
        self._num_stochastic_matrix_entries = None
        self._num_stochastic_costs = None

        self._num_first_stage_constraints = None
        self._num_second_stage_constraints = None

        # If this constraint appears in the LP file it means the
        # objective included constant terms
        self._count_constraint_ONE_VAR_CONSTANT = 0

        self._AllVars = []

        # Read from siphelp output. Includes cplex row, column
        # assignments Required for stochastic matrix file
        self._ColumnMap = {}
        self._ColumnNameList = []
        self._RowMap = {}
        self._RowNameList = []

        self._reference_scenario_instance = self._reference_scenario._instance

    def write(self, ph, global_reference_scenario=False):

        scenario_instance = self._reference_scenario_instance

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

        if global_reference_scenario:
            assert os.path.exists(self._lpfilename)
            try:
                os.remove(os.path.join(os.getcwd(),_ddsip_help_output))
            except OSError:
                pass
            try:
                os.remove(os.path.join(os.getcwd(),_ddsip_help_output+".gz"))
            except OSError:
                pass
            assert not os.path.exists(os.path.join(os.getcwd(),_ddsip_help_output))
            assert not os.path.exists(os.path.join(os.getcwd(),_ddsip_help_output+".gz"))
            print(("COMMAND= "+str(_ddsip_help)+' '+self._lpfilename+' '+str(max_name_len)))
            os.system(_ddsip_help+' '+self._lpfilename+' '+str(max_name_len))
            assert os.path.exists(os.path.join(os.getcwd(),_ddsip_help_output+".gz"))
            os.system("gzip -df "+os.path.join(os.getcwd(),_ddsip_help_output+".gz"))
        else:
            assert os.path.exists(os.path.join(os.getcwd(),_ddsip_help_output))

        dashcount = 0
        varcount = 0
        concount = 0
        with open(_ddsip_help_output,'r') as f:
            for line in f.readlines():
                pieces = line.split()
                if dashcount == 3:
                    break
                elif pieces[0] =='__________________________':
                    dashcount = dashcount+1
                elif (dashcount == 1) and (len(pieces) == 2):
                    varcount += 1
                    self._ColumnMap[pieces[1]] = int(pieces[0])
                    self._ColumnNameList.append(pieces[1])
                    assert len(self._ColumnNameList)-1 == \
                        self._ColumnMap[self._ColumnNameList[-1]]
                elif (dashcount == 2) and (len(pieces) == 2):
                    concount += 1
                    self._RowMap[pieces[1]] = int(pieces[0])
                    self._RowNameList.append(pieces[1])
                    assert len(self._RowNameList)-1 == \
                        self._RowMap[self._RowNameList[-1]]

        (ObjObject,
         ConstraintMap,
         remaining_lpfile_rows,
         AllConstraintNames) = \
            self._read_parse_lp(self._lpfilename)

        os.remove(self._lpfilename)

        #print("@@@@@@@@@@@ COLS @@@@@@@@@@@@")
        #print(set(self._ColumnNameList)-set(self._AllVars))
        #print("@@@@@@@@@@@ ROWS @@@@@@@@@@@@")
        #print(set(self._RowNameList)-set(AllConstraintNames))

        # ONE_VAR_CONSTANT always appears as an extra variable in the LP file
        assert len(self._ColumnMap)-1 == len(self._AllVars)

        # c_e_ONE_VAR_CONSTANT only appears as an extra constraint when
        # the objective function has a constant term
        assert len(self._RowMap) == \
            len(AllConstraintNames) + \
            self._count_constraint_ONE_VAR_CONSTANT
        #print sorted(ObjObject.VarToCoeff.keys())
        #print [name for name, col in \
        #       sorted(self._ColumnMap.items(),key=itemgetter(1))]
        assert set(ObjObject.VarToCoeff.keys()) == \
            set([name for name, col in \
                 sorted(list(self._ColumnMap.items()),
                        key=itemgetter(1))][:len(ObjObject.VarToCoeff)])

        if global_reference_scenario:
            (MatrixEntries_ConstrToRow_Map,
             SecondStageConstrOrder,
             self._num_first_stage_constraints,
             self._num_second_stage_constraints) = \
                self.sort_write_lp(ph,
                                   ObjObject,
                                   ConstraintMap,
                                   StageToConstraintMap,
                                   remaining_lpfile_rows)
        else:
            # for the matrix.sc file we need to know which constraint is
            # in which row:
            MatrixEntries_ConstrToRow_Map = {}
            FirstStage = StageToConstraintMap['FirstStage']
            ConstrNames = set(ConstraintMap.keys())
            FirstStageConstrOrder = sorted(ConstrNames.intersection(FirstStage))
            SecondStageConstrOrder = sorted(ConstrNames - set(FirstStage))

            # so that we know in which rows the constraints with
            # stochastic data are (first row has index 0)
            self._num_first_stage_constraints = 0
            self._num_second_stage_constraints = 0
            count_rows = -1
            for name in FirstStageConstrOrder:
                count_rows += 1
                MatrixEntries_ConstrToRow_Map[name] = count_rows
                self._num_first_stage_constraints += 1
            for name in SecondStageConstrOrder:
                count_rows += 1
                MatrixEntries_ConstrToRow_Map[name] = count_rows
                self._num_second_stage_constraints += 1

        if not getattr(self._reference_scenario_instance,
                       _disable_stoch_rhs_flagname,
                       _disable_stoch_rhs_default):
            self._num_stochastic_rhs = 0
            self._write_rhs_sc(ph,
                               ConstraintMap,
                               SecondStageConstrOrder)

        if not getattr(self._reference_scenario_instance,
                       _disable_stoch_costs_flagname,
                       _disable_stoch_costs_default):
            self._num_stochastic_costs = 0
            self._write_obj_sc(ph, ObjObject)

        if not getattr(self._reference_scenario_instance,
                       _disable_stoch_matrix_flagname,
                       _disable_stoch_matrix_default):
            self._num_stochastic_matrix_entries = 0
            self._write_matrix_sc(
                ph,
                ConstraintMap,
                SecondStageConstrOrder,
                MatrixEntries_ConstrToRow_Map,
                include_position_section=global_reference_scenario)

        if global_reference_scenario:
            return (self.input_file_name_list,
                    self._FirstStageVars,
                    self._FirstStageVarIdMap,
                    self._FirstStageDerivedVars,
                    self._FirstStageDerivedVarIdMap,
                    self._SecondStageVars,
                    self._SecondStageVarIdMap,
                    self._StageCostVarStageMap,
                    self._num_stochastic_rhs,
                    self._num_stochastic_matrix_entries,
                    self._num_stochastic_costs,
                    self._num_first_stage_constraints,
                    self._num_second_stage_constraints,
                    self._count_constraint_ONE_VAR_CONSTANT)

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
            set(id(vardata) for scenariotree_id, vardata \
                    in iteritems(scenariotree_vars) \
                    if scenariotree_id in rootnode_vars)
        capabilities = lambda x: True
        text_labeler = TextLabeler()
        labeler = lambda x: text_labeler(x) + \
                  (""
                   if ((not isinstance(x, _VarData)) or \
                       (id(x) not in firststage_ids)) else \
                   self._firststage_var_suffix)
        output_filename, symbol_map = \
            lp_file_writer(self._reference_scenario_instance,
                           self._lpfilename,
                           capabilities,
                           {'labeler' : labeler})
        lp_file_writer.deactivate()
        assert output_filename == self._lpfilename
        #print("\nModel written\n")

        # Return the max symbol length (for siphelp)
        max_name_len = max(len(symbol) for symbol in symbol_map.bySymbol)
        return max_name_len, symbol_map

    def sort_write_lp(self, ph, ObjObject, ConstraintMap, StageToConstraintMap, remaining_lpfile_rows):

        try:
            #print("\nWrite the LP file for dd in sorted_LPfile.lp\n")
            lp = open("sorted_LPfile.lp", "wt")
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
        ConstrNames = set(ConstraintMap.keys())
        FirstStageConstrOrder = sorted(ConstrNames.intersection(FirstStage))
        SecondStageConstrOrder = sorted(ConstrNames - set(FirstStage))

        # so that we know in which rows the constraints with
        # stochastic data are (first row has index 0)
        count_rows = -1

        self._num_first_stage_constraints = 0
        self._num_second_stage_constraints = 0
        for name in FirstStageConstrOrder:
            lp.write("\n"+name+":\n")
            count_rows += 1
            MatrixEntries_ConstrToRow_Map[name] = count_rows
            lp_con = ConstraintMap[name]
            self.print_coeff_var_from_map(lp_con[0], lp)
            lp.write(lp_con[1]+" "+lp_con[2]+"\n")
            self._num_first_stage_constraints += 1

        for name in SecondStageConstrOrder:
            lp.write("\n"+name+":\n")
            count_rows += 1
            MatrixEntries_ConstrToRow_Map[name] = count_rows
            lp_con = ConstraintMap[name]
            self.print_coeff_var_from_map(lp_con[0], lp)
            lp.write(lp_con[1]+" "+lp_con[2]+"\n")
            self._num_second_stage_constraints += 1

        # print the remaining rows of the lp file
        for row in range(len(remaining_lpfile_rows)):
            lp.write("\n")
            for i in range(len(remaining_lpfile_rows[row])):
                lp.write(remaining_lpfile_rows[row][i]+" ")

        lp.close()

        return MatrixEntries_ConstrToRow_Map, SecondStageConstrOrder, self._num_first_stage_constraints, self._num_second_stage_constraints

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
            #print(("\nName of the lp file which will be read: "+str(lp_filename)+"\n"))
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

            while list_lp_file[CurrRow][0].strip() != 'bounds':

                # If this constraint appears in the LP file it means the
                # objective included constant terms
                if "c_e_ONE_VAR_CONSTANT" in list_lp_file[CurrRow][0]:
                    self._count_constraint_ONE_VAR_CONSTANT = 1
                    break

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
                    # constraint (AssignComparator and AssignRHS need them)
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

        if self._count_constraint_ONE_VAR_CONSTANT:
            # if this constraint appears make sure it appears as
            # the last constraint in the LP file and will be
            # included with the remaining_lpfile_rows
            assert "c_e_ONE_VAR_CONSTANT" in list_lp_file[CurrRow][0]
            assert CurrRow < len(list_lp_file)

        remaining_lpfile_rows = []
        while CurrRow < len(list_lp_file):
            remaining_lpfile_rows.append(list_lp_file[CurrRow])
            CurrRow += 1
        AllConstraintNames = list(ConstraintMap.keys())

        return ObjObject, ConstraintMap, remaining_lpfile_rows, AllConstraintNames

    def print_coeff_var_from_map(self, VarToCoeffMap, lp_file):
        keys = list(VarToCoeffMap.keys())
        #string sorting
        keys.sort()
        line_template = "%+"+_precision_string+" %s\n"
        for key in keys:
            lp_file.write(line_template % (VarToCoeffMap[key], key))

    def SkipLPNullLines(self, CurrRow, lp_file):
        # skip over comment lines and blank lines (assuming comment
        # character leads)
        while (lp_file[CurrRow] == [] or lp_file[CurrRow][0].startswith("\\")):
            CurrRow += 1
            if CurrRow >= len(lp_file):
                print("Error: past end of file while skipping null lines")
                sys.exit(1)
        return CurrRow

    def ClearName(self, name):
        if name[-1] == ':':
            name = name[:-1]
        return name

    def LocalPostionUpdate(self, lp_file, CurrRow, CurrElem, step):
        if ((step !=1) and (step != 2)):
            print(("unexpected step length  ="+str(step)))
        CurrElem += step
        if CurrElem > len(lp_file[CurrRow])-1:
            CurrRow += 1
            CurrElem = 0
        return CurrRow, CurrElem

    def _Populate_StageVars(self, ph, LP_symbol_map):

        all_vars_cnt = 0
        piecewise_blocks = []
        for block in self._reference_scenario_instance.block_data_objects(active=True):
            all_vars_cnt += len(list(block.component_data_objects(Var, descend_into=False)))
            if isinstance(block, (Piecewise, _PiecewiseData)):
                piecewise_blocks.append(block)

        rootnode = ph._scenario_tree.findRootNode()

        stagetwo = ph._scenario_tree._stages[1]
        leafnode = self._reference_scenario._leaf_node
        LP_byObject = LP_symbol_map.byObject
        for scenario_tree_id, vardata in \
              iteritems(self._reference_scenario_instance.\
              _ScenarioTreeSymbolMap.bySymbol):
            if vardata.is_expression():
                continue
            try:
                LP_name = LP_byObject[id(vardata)]
            except KeyError:
                raise ValueError("Variable with name '%s' was declared "
                                 "on the scenario tree but did not appear "
                                 "in the reference scenario LP file."
                                 % (vardata.name))
            if scenario_tree_id in rootnode._standard_variable_ids:
                self._FirstStageVars.append(LP_name)
                self._FirstStageVarIdMap[LP_name] = scenario_tree_id
            elif (scenario_tree_id in rootnode._derived_variable_ids):
                self._FirstStageDerivedVars.append(LP_name)
                self._FirstStageDerivedVarIdMap[LP_name] = scenario_tree_id
            elif (scenario_tree_id in leafnode._variable_ids):
                self._SecondStageVars.append(LP_name)
                self._SecondStageVarIdMap[LP_name] = scenario_tree_id
            else:
                print(("%s %s" % (str(scenario_tree_id), str(vardata.name))))
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
                    self._StageCostVarStageMap[LP_name] = stage._name
                    self._AllVars.append(LP_name)

        # The *ONLY* case where we allow variables to exist on the
        # model that were not declared on the scenario tree is when
        # they are autogenerated by a Piecewise component

        # For now we just assume all auxiliary Piecewise variables
        # are SecondStage
        for block in piecewise_blocks:
            for vardata in block.component_data_objects(Var,
                                                        active=True,
                                                        descend_into=False):
                LP_name = LP_byObject[id(vardata)]
                self._SecondStageVars.append(LP_name)
                self._SecondStageVarIdMap[LP_name] = scenario_tree_id
                self._AllVars.append(LP_name)

        # Make sure every variable on the model has been
        # declared on the scenario tree
        if len(self._AllVars) != all_vars_cnt:
            print("**** THERE IS A PROBLEM ****")
            print("Not all model variables are on the scenario tree. Investigating...")
            print("len(self._AllVars)=", len(self._AllVars), "all_vars_cnt=", all_vars_cnt)
            all_vars = set()
            tmp_buffer = {}
            for block in self._reference_scenario_instance.block_data_objects(active=True):
                all_vars.update(vardata.getname(True, tmp_buffer) \
                                for vardata in block.component_data_objects(Var, descend_into=False))
            print(("Number of Variables Found on Model: "+str(len(all_vars))))
            print ("writing all_vars.dat")
            with open("allvars.dat",'w') as f:
                f.write("allvars.dat\n")
                for av in all_vars:
                    f.write((str(av)+"\n"))
            tree_vars = set()
            for scenario_tree_id, vardata in \
                iteritems(self._reference_scenario_instance.\
                          _ScenarioTreeSymbolMap.bySymbol):
                tree_vars.add(vardata.name)
            print(("Number of Scenario Tree Variables (found in ddsip LP file): "+str(len(tree_vars))))
            print ("writing tree_vars.dat")
            with open("tree_vars.dat",'w') as f:
                f.write("tree_vars.dat\n")
                for tv in tree_vars:
                    f.write((str(tv)+"\n"))
            cost_vars = set()
            for stage in ph._scenario_tree._stages:
                cost_variable_name, cost_variable_index = \
                    stage._cost_variable
                stage_cost_component = \
                    self._reference_scenario_instance.\
                    find_component(cost_variable_name)
                if stage_cost_component.type() is not Expression:
                    cost_vars.add(stage_cost_component[cost_variable_index].name)
            print(("Number of Scenario Tree Cost Variables (found in ddsip LP file): "+str(len(cost_vars))))
            print ("writing cost_vars.dat")
            with open("cost_vars.dat","w") as f:
                f.write("cost_vars.dat\n")
                for cv in cost_vars:
                    f.write((str(cv)+"\n"))
            print("Variables Missing from Scenario Tree (or LP file):")
            MissingSet = all_vars-(tree_vars+cost_vars)
            for ims in MissingSet:
                print ("    ",ims)
            raise ValueError("Missing scenario tree variable declarations")

        # A necessary but not sufficient sanity check to make sure the
        # second stage variable sets are the same for all
        # scenarios. This is not required by pysp, but I think this
        # assumption is made in the rest of the code here
        for tree_node in stagetwo._tree_nodes:
            assert len(leafnode._variable_ids) == \
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
            block_canonical_repn = getattr(block, "_canonical_repn", None)
            if block_canonical_repn is None:
                raise ValueError("Unable to find _canonical_repn ComponentMap "
                                 "on block %s" % (block.name))
            isPiecewise = False
            if isinstance(block, (Piecewise, _PiecewiseData)):
                isPiecewise = True
            for constraint_data in block.component_data_objects(
                    SOSConstraint,
                    active=True,
                    descend_into=False):
                raise TypeError("SOSConstraints are not handled by the "
                                "DDSIP interface: %s"
                                % (constraint_data.name))
            for constraint_data in block.component_data_objects(
                    Constraint,
                    active=True,
                    descend_into=False):
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

    def _write_rhs_sc(self, ph, ConstraintMap, SecondStageConstrOrder):

        with open(self._rhsfilename, 'wt') as f:

            if isinstance(ph, _PHSolverServer):
                probability = ph._uncompressed_scenario_tree.get_scenario(
                    self._reference_scenario._name)._probability
            else:
                probability = self._reference_scenario._probability
            header = ("scenario"+str(self._scenario_index)+"\n"
                      +str(probability)+"\n")
            f.write(header)

            self._num_stochastic_rhs = 0
            for name in SecondStageConstrOrder:
                f.write((("%"+_precision_string+"\n")
                         % (float(ConstraintMap[name][2]))))
                self._num_stochastic_rhs += 1

    def _write_obj_sc(self, ph, ObjObject):

        with open(self._objfilename, 'wt') as f:

            f.write("scenario"+str(self._scenario_index)+"\n")
            self._num_stochastic_costs = 0
            varnames_appearing = sorted(ObjObject.VarToCoeff.keys())
            for name in self._ColumnMap:
                if name not in self._FirstStageVars:
                    if name in varnames_appearing:
                        f.write(("%"+_precision_string+"\n")
                                % (ObjObject.VarToCoeff[name]))
                    else:
                        f.write("0\n")
                    self._num_stochastic_costs += 1

    def _write_matrix_sc(self,
                         ph,
                         ConstraintMap,
                         SecondStageConstrOrder,
                         MatrixEntries_ConstrToRow_Map,
                         include_position_section=False):

        with open(self._matfilename, 'wt') as f:

            if include_position_section:
                f.write('position\n')
                for conname in SecondStageConstrOrder:
                    # either it follows with what ddsip help writes in
                    # the rows+cols files
                    row = self._RowMap[conname]
                    # or it follows our ordering based off of
                    # stochastic rows following first-stage rows
                    ###row = MatrixEntries_ConstrToRow_Map[conname]
                    # or something else entirely ?????

                    var_coeffs = ConstraintMap[conname].VarToCoeff
                    for varname in sorted(var_coeffs):
                        f.write("%s\n%s\n"
                                % (row,
                                   self._ColumnMap[varname]))

            self._num_stochastic_matrix_entries = 0
            f.write("scenario"+str(self._scenario_index)+"\n")
            for conname in SecondStageConstrOrder:
                # either it follows with what ddsip ddhelp writes in
                # the rows+cols files
                row = self._RowMap[conname]
                # or it follows our ordering based off of
                # stochastic rows following first-stage
                ###row = MatrixEntries_ConstrToRow_Map[conname]
                # or something else entirely ?????

                var_coeffs = ConstraintMap[conname].VarToCoeff
                for varname in sorted(var_coeffs):
                    f.write(("%"+_precision_string+"\n")
                            % (var_coeffs[varname]))
                    self._num_stochastic_matrix_entries += 1
            print(("%s %s" % list(map(str,(self._reference_scenario._name, self._num_stochastic_matrix_entries)))))

class ddextension(pyomo.util.plugin.SingletonPlugin):

    pyomo.util.plugin.implements(phextension.IPHExtension)

    def __init__(self):

        self._lpfilename = "ddlpfile.lp"
        self._firststage_var_suffix = '__DDSIP_FIRSTSTAGE'
        self._reference_scenario = None

        # keep track of input file names
        self.input_file_name_list = []
        self._FirstStageVars = []
        self._FirstStageVarIdMap = {}
        self._FirstStageDerivedVars = []
        self._FirstStageDerivedVarIdMap = {}
        self._SecondStageVars = []
        self._SecondStageVarIdMap = {}
        self._StageCostVarStageMap = {}
        self._num_stochastic_rhs = None
        self._num_stochastic_matrix_entries = None
        self._num_stochastic_costs = None
        self._num_first_stage_constraints = None
        self._num_second_stage_constraints = None
        self._count_constraint_ONE_VAR_CONSTANT = 0

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
        reference_scenario_name = self._reference_scenario._name

        if ph._scenario_tree.contains_bundles():
            print("** The DDSIP interface is ignoring scenario bundles **")

        print(("\nUsing %s as reference scenario" % (reference_scenario_name)))

        if isinstance(ph._solver_manager,
                      pyomo.solvers.plugins.smanager.phpyro.SolverManager_PHPyro):

            if ph._scenario_tree.contains_bundles():

                scenario_to_bundle_map = {}
                for scenario_index, scenario_name in enumerate(self._ScenarioVector, 1):
                    for bundle in ph._scenario_tree._scenario_bundles:
                        if scenario_name in bundle._scenario_names:
                            scenario_to_bundle_map[scenario_name] = bundle._name

                ah = transmit_external_function_invocation_to_worker(
                    ph,
                    scenario_to_bundle_map[reference_scenario_name],
                    thisfile,
                    "Write_DDSIP_Input",
                    function_args=(reference_scenario_name,
                                   self._ScenarioVector.index(reference_scenario_name)+1,
                                   self._firststage_var_suffix),
                    function_kwds={'global_reference_scenario':True},
                    return_action_handle=True)
                ph._solver_manager.wait_all([ah])
                (self.input_file_name_list,
                 self._FirstStageVars,
                 self._FirstStageVarIdMap,
                 self._FirstStageDerivedVars,
                 self._FirstStageDerivedVarIdMap,
                 self._SecondStageVars,
                 self._SecondStageVarIdMap,
                 self._StageCostVarStageMap,
                 self._num_stochastic_rhs,
                 self._num_stochastic_matrix_entries,
                 self._num_stochastic_costs,
                 self._num_first_stage_constraints,
                 self._num_second_stage_constraints,
                 self._count_constraint_ONE_VAR_CONSTANT) = \
                    ph._solver_manager.get_results(ah)

                assert os.path.exists(os.path.join(os.getcwd(),'rows+cols'))
                ahs = []
                for scenario_index, scenario_name in enumerate(self._ScenarioVector, 1):
                    if scenario_name != reference_scenario_name:
                        ahs.append(transmit_external_function_invocation_to_worker(
                            ph,
                            scenario_to_bundle_map[scenario_name],
                            thisfile,
                            "Write_DDSIP_Input",
                            function_args=(scenario_name,
                                           scenario_index,
                                           self._firststage_var_suffix),
                            function_kwds={'global_reference_scenario':False},
                            return_action_handle=True))
                ph._solver_manager.wait_all(ahs)

            else:

                ah = transmit_external_function_invocation_to_worker(
                    ph,
                    reference_scenario_name,
                    thisfile,
                    "Write_DDSIP_Input",
                    function_args=(reference_scenario_name,
                                   self._ScenarioVector.index(reference_scenario_name)+1,
                                   self._firststage_var_suffix),
                    function_kwds={'global_reference_scenario':True},
                    return_action_handle=True)
                ph._solver_manager.wait_all([ah])
                (self.input_file_name_list,
                 self._FirstStageVars,
                 self._FirstStageVarIdMap,
                 self._FirstStageDerivedVars,
                 self._FirstStageDerivedVarIdMap,
                 self._SecondStageVars,
                 self._SecondStageVarIdMap,
                 self._StageCostVarStageMap,
                 self._num_stochastic_rhs,
                 self._num_stochastic_matrix_entries,
                 self._num_stochastic_costs,
                 self._num_first_stage_constraints,
                 self._num_second_stage_constraints,
                 self._count_constraint_ONE_VAR_CONSTANT) = \
                    ph._solver_manager.get_results(ah)

                assert os.path.exists(os.path.join(os.getcwd(),'rows+cols'))
                ahs = []
                for scenario_index, scenario_name in enumerate(self._ScenarioVector, 1):
                    if scenario_name != reference_scenario_name:
                        ahs.append(transmit_external_function_invocation_to_worker(
                            ph,
                            scenario_name,
                            thisfile,
                            "Write_DDSIP_Input",
                            function_args=(scenario_name,
                                           scenario_index,
                                           self._firststage_var_suffix),
                            function_kwds={'global_reference_scenario':False},
                            return_action_handle=True))
                ph._solver_manager.wait_all(ahs)
        else:
            (self.input_file_name_list,
             self._FirstStageVars,
             self._FirstStageVarIdMap,
             self._FirstStageDerivedVars,
             self._FirstStageDerivedVarIdMap,
             self._SecondStageVars,
             self._SecondStageVarIdMap,
             self._StageCostVarStageMap,
             self._num_stochastic_rhs,
             self._num_stochastic_matrix_entries,
             self._num_stochastic_costs,
             self._num_first_stage_constraints,
             self._num_second_stage_constraints,
             self._count_constraint_ONE_VAR_CONSTANT) = \
                Write_DDSIP_Input(ph, ph._scenario_tree, self._reference_scenario,
                                  reference_scenario_name,
                                  self._ScenarioVector.index(reference_scenario_name)+1,
                                  self._firststage_var_suffix,
                                  global_reference_scenario=True)

            for scenario_index, scenario_name in enumerate(self._ScenarioVector, 1):
                if scenario_name != reference_scenario_name:
                    Write_DDSIP_Input(ph, ph._scenario_tree, ph._scenario_tree.get_scenario(scenario_name),
                                      scenario_name,
                                      scenario_index,
                                      self._firststage_var_suffix,
                                      global_reference_scenario=False)

        if self._num_stochastic_rhs is None:
            print("")
            print(("Detected "+_disable_stoch_rhs_flagname+"=True"))
            print("Constraint rhs entries are assumed to be deterministic.")
        else:
            assert self._num_stochastic_rhs > 0
            rhs_filename = 'rhs.sc'
            RHS_file = open(rhs_filename,'wt')
            RHS_file.close()
            assert self._ScenarioVector[0] == reference_scenario_name
            os.system('cat '+rhs_filename+"."+reference_scenario_name+" >> "+rhs_filename)
            os.remove(rhs_filename+"."+reference_scenario_name)
            for scenario_name in self._ScenarioVector[1:]:
                assert os.path.exists(rhs_filename+"."+scenario_name)
                os.system('cat '+rhs_filename+"."+scenario_name+" >> "+rhs_filename)
                os.remove(rhs_filename+"."+scenario_name)
            # keep track of input file names
            self.input_file_name_list.append(rhs_filename)

        if self._num_stochastic_costs is None:
            print("")
            print(("Detected "+_disable_stoch_costs_flagname+"=True"))
            print("Cost terms are assumed to be deterministic.")
        else:
            assert self._num_stochastic_costs > 0
            obj_filename = 'cost.sc'
            OBJ_file = open(obj_filename,'wt')
            OBJ_file.close()
            assert self._ScenarioVector[0] == reference_scenario_name
            os.system('cat '+obj_filename+"."+reference_scenario_name+" >> "+obj_filename)
            os.remove(obj_filename+"."+reference_scenario_name)
            for scenario_name in self._ScenarioVector[1:]:
                assert os.path.exists(obj_filename+"."+scenario_name)
                os.system('cat '+obj_filename+"."+scenario_name+" >> "+obj_filename)
                os.remove(obj_filename+"."+scenario_name)
            # keep track of input file names
            self.input_file_name_list.append(obj_filename)

        if self._num_stochastic_matrix_entries is None:
            print("")
            print(("Detected "+_disable_stoch_matrix_flagname+"=True"))
            print("Constraint matrix entries are assumed to be deterministic.")
        else:
            assert self._num_stochastic_matrix_entries > 0
            mat_filename = 'matrix.sc'
            MAT_file = open(mat_filename,'wt')
            MAT_file.close()
            assert self._ScenarioVector[0] == reference_scenario_name
            os.system('cat '+mat_filename+"."+reference_scenario_name+" >> "+mat_filename)
            os.remove(mat_filename+"."+reference_scenario_name)
            for scenario_name in self._ScenarioVector[1:]:
                assert os.path.exists(mat_filename+"."+scenario_name)
                os.system('cat '+mat_filename+"."+scenario_name+" >> "+mat_filename)
                os.remove(mat_filename+"."+scenario_name)
            # keep track of input file names
            self.input_file_name_list.append(mat_filename)

        self.write_sip_in(ph)
        self.write_input2sip(ph)
        print("ddextension post_ph_initialization callback done")

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

        self.write_start_in(ph)
        # Write this file again now that we have weights
        # and variable starts
        self.write_input2sip(ph)
        #sipout = 'sipout_good'
        #os.system('rm -rf '+sipout)
        #os.system(_ddsip_exe+' < 2sip')
        #assert os.path.exists(sipout) and os.path.isdir(sipout)
        #assert os.path.exists(os.path.join(sipout,'solution.out'))
        #self._load_ddsip_solution(ph, os.path.join(sipout,'solution.out'))

    def write_sip_in(self, ph):
        try:
            print("\n\nWrite dd input file: sip.in \n")
            sipin = open("sip.in", "wt")
        except IOError:
            print("sip.in cannot be created!")
            sys.exit(1)

        # keep track of input file names
        self.input_file_name_list.append('sip.in')

        NumberOfFirstStageVars = len(self._FirstStageVars)
        # Account for the extra ONE_VAR_CONSTANT variable that
        # shows up in every Pyomo LP file
        NumberOfSecondStageVars = (len(self._SecondStageVars) + \
                                   len(self._FirstStageDerivedVars) + \
                                   1)

        NumberOfStochasticRHS = self._num_stochastic_rhs
        if NumberOfStochasticRHS is None:
            NumberOfStochasticRHS = 0

        NumberOfStochasticCosts = self._num_stochastic_costs
        if NumberOfStochasticCosts is None:
            NumberOfStochasticCosts = 0

        NumberOfStochasticMatrixEntries = self._num_stochastic_matrix_entries
        if NumberOfStochasticMatrixEntries is None:
            NumberOfStochasticMatrixEntries = 0

        NumberOfFirstStageConstraints = self._num_first_stage_constraints
        # Account for the c_e_ONE_VAR_CONSTANT constraint if it appeared
        # (i.e., the objective had a constant term)
        NumberOfSecondStageConstraints = (self._num_second_stage_constraints + \
                                          self._count_constraint_ONE_VAR_CONSTANT)
        
        NumberOfScenarios = len(self._ScenarioVector)

        sipin.write('BEGIN \n\n\n')
        sipin.write('FIRSTCON '+str(NumberOfFirstStageConstraints)+'\n')
        sipin.write('FIRSTVAR '+str(NumberOfFirstStageVars)+'\n')
        sipin.write('SECCON '+str(NumberOfSecondStageConstraints)+'\n')
        sipin.write('SECVAR '+str(NumberOfSecondStageVars)+'\n')
        sipin.write('POSTFIX '+self._firststage_var_suffix+'\n')
        sipin.write('SCENAR '+str(NumberOfScenarios)+'\n')

        sipin.write('STOCRHS '+str(NumberOfStochasticRHS)+'\n')
        sipin.write('STOCCOST '+str(NumberOfStochasticCosts)+'\n')
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

    def write_start_in(self, ph):
        ### Note: as of Dec 26, 2014, the ph object we have here does not have the
        ###       scenario tree that has the solution, so this will not work
        ###       I put this code in a solution writer to get what I need - DLW
        # keep track of input file names -- or not (dlw Dec 2014)
        ### self.input_file_name_list.append('start.in')

        # keep track of input file names
        self.input_file_name_list.append('start.in')

        print("\n\nWrite dd input file: start.in (** integers will be rounded ** )\n")

        # assume two-stage and we only want the first stage vars
        rootnode = ph._scenario_tree.findRootNode()
        """
        # note: DDSIP seems to get the names in alpha order from cplex
        VNames = []
        VVals = {}
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
        with open('start.in', 'wt') as f:
           f.write("SOLUTION\n")
           for name in VNames:
                if rootnode.is_variable_discrete(variable_id):
                    val = str(round(float(VVals[name])))
                else:
                    val = str(float(VVals[name]))
                f.write(val+"\n")
                # aside: rounding changes slightly from Python 2.x to 3.x
        """

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
        num_scenarios = float(len(self._ScenarioVector))

        with open('PHWEIGHTS.csv','wt') as f:
            f.write("varname,"+(",".join(name for name in self._ScenarioVector))+"\n")
            for var_index, vector_w in enumerate(weights_vectors):
                varname = self._FirstStageVars[var_index]
                f.write(varname+","+(",".join(repr(w) for w in vector_w))+"\n")

        with open('start.in','wt') as f:
            f.write("SOLUTION\n")
            for name in self._FirstStageVars:
                scenario_tree_id = self._FirstStageVarIdMap[name]
                if rootnode.is_variable_discrete(scenario_tree_id):
                    f.write(("%"+_precision_string+"\n")
                            % (round(rootnode._xbars[scenario_tree_id])))
                else:
                    f.write(("%"+_precision_string+"\n")
                            % (rootnode._xbars[scenario_tree_id]))

            # keep track of input file names
            ###self.input_file_name_list.append('NONANT1.in')

            # NONANT 1 case:
            v1_transpose = []
            for vector_w in weights_vectors:
                # all but the first position
                vector_w_iter = vector_w[1:]
                tmp = [-w/num_scenarios \
                       for w in vector_w_iter]
                v1_transpose.append(tmp)
            with open('NONANT1.in','wt') as fp:
                fp.write('MULTIPLIER\n')
                f.write('MULTIPLIER\n')
                # transpose and flatten
                for column in zip(*v1_transpose):
                    fp.writelines(repr(x)+'\n' for x in column)
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

    def _load_ddsip_solution(self, ph, filename):

        print("Loading DDSIP solution into scenario tree")

        # If stage costs are variables, then we will be able to
        # extract them from the DDSIP solution otherwise we will have
        # to evaludate the Expression component on the instances. If
        # this is parallel ph the instances will not be present so we
        # will not be able to determine individual stage costs.  In
        # either case, set them to None now just in case we can obtain
        # values for them.
        for scenario in ph._scenario_tree._scenarios:
            for stage_name in scenario._stage_costs:
                scenario._stage_costs[stage_name] = None

        root_node_name = ph._scenario_tree.findRootNode()._name
        reference_leaf_node = self._reference_scenario._leaf_node
        with open(filename) as f:
            line = f.readline()
            while line:
                # Get first stage solution
                if "1. Best Solution" in line:
                    first_stage_solution = {}
                    assert "Variable name                Value" in f.readline()
                    assert "---" in f.readline()
                    for i in range(len(self._FirstStageVarIdMap)):
                        LP_name, sol = f.readline().strip().split()
                        first_stage_solution[self._FirstStageVarIdMap[LP_name]] = float(sol)
                    for scenario in ph._scenario_tree._scenarios:
                        scenario._x[root_node_name].update(first_stage_solution)
                    f.readline()
                elif "2. Bounds" in line:
                    assert "Scenario    Lower Bound (root)    Upper Bound" in f.readline()
                    assert "---" in f.readline()
                    for i in range(len(ph._scenario_tree._scenarios)):
                        scenario_id, obj_lb, obj_ub = f.readline().strip().split()
                        scenario = ph._scenario_tree.get_scenario(self._ScenarioVector[int(scenario_id)-1])
                        scenario._objective = float(obj_ub)
                        scenario._cost = float(obj_ub)
                    f.readline()
                elif "3. Quantiles" in line:
                    # We don't use any information from this section
                    while f.readline().strip():
                        pass
                elif "4. Second-stage solutions" in line:
                    for i in range(len(ph._scenario_tree._scenarios)):
                        scenario_id = int(f.readline().strip().split()[1][:-1])
                        scenario = ph._scenario_tree.get_scenario(self._ScenarioVector[scenario_id-1])
                        scenario_firststage_solution = scenario._x[root_node_name]
                        secondstage_solution = []
                        # Add 1 for ONE_VAR_CONSTANT
                        for j in range(len(self._SecondStageVars)+1):
                            LP_name, sol = f.readline().strip().split()
                            if LP_name in self._SecondStageVarIdMap:
                                secondstage_solution.append((self._SecondStageVarIdMap[LP_name],float(sol)))
                            elif LP_name in self._FirstStageDerivedVarIdMap:
                                scenario_firststage_solution[self._FirstStageDerivedVarIdMap[LP_name]] = float(sol)
                            elif LP_name in self._StageCostVarStageMap:
                                scenario._stage_costs[self._StageCostVarStageMap[LP_name]._name] = float(sol)
                            else:
                                assert LP_name == "ONE_VAR_CONSTANT"
                        leaf_node = scenario._leaf_node
                        print(("%s %s" % list(map(str,(leaf_node._name, reference_leaf_node._name)))))
                        scenario_secondstage_solution = scenario._x[leaf_node._name]
                        for reference_node_variable_id, sol in secondstage_solution:
                            this_node_variable_id = leaf_node._name_index_to_id[reference_leaf_node._variable_ids[reference_node_variable_id]]
                            scenario_secondstage_solution[this_node_variable_id] = sol

                line = f.readline()

        print("Updating variable statistics after loading DDSIP solution")
        ph.update_variable_statistics()

        # If the scenario instances are present, push the
        # solutions from the scenario tree. Also try to
        # recover the individual stage costs if these
        # were not variables in the model (and DDSIP solution)
        for scenario in ph._scenario_tree._scenarios:
            if scenario._instance is not None:
                print(("%s %s" % list(map(str(scenario._name, scenario._instance.name)))))
                scenario.push_solution_to_instance()
                scenario.update_solution_from_instance()
                """
                for tree_node in scenario._node_list:
                    stage_name = tree_node._stage._name
                    if scenario._stage_costs[stage_name] is None:
                        cost_variable_name, cost_variable_index = \
                            tree_node._stage._cost_variable
                        stage_cost_component = scenario._instance.find_component(cost_variable_name)
                        scenario._stage_costs[stage_name] = \
                        stage_cost_component[cost_variable_index](exception=False)
                """
        warn = False
        for scenario in ph._scenario_tree._scenarios:
            print(("%s %s" % list(map(str,(scenario._name, scenario._cost)))))
            for stage_name in scenario._stage_costs:
                if scenario._stage_costs[stage_name] is None:
                    warn = True
                    break
        if warn:
            print("***WARNING*** Individual stage costs could not be "
                  "recovered from DDSIP solution. Try adding an artificial "
                  "variable to the model for each stage cost and assign this "
                  "to the stage cost in the scenario tree specification file.")

#
# Helper Classes
#

class MatrixEntriesClass(object):
    __slots__ = ('VarUniqueValueMap',)

    def __init__(self):
        # key is Variable name, value is unique number (keeps track of
        # stochastic data)
        self.VarUniqueValueMap = {}

    def __call__(self):
        return self.VarUniqueValueMap

    def __getitem__(self, index):
        if index == 0:
            return self.VarUniqueValueMap

    def AddToMap(self, file_line):
        var = file_line[2]
        uniquenumber = file_line[3]
        self.VarUniqueValueMap[var] = uniquenumber
        #print "in MatrixEntriesClass: self.VarUniqueValueMap=", self.VarUniqueValueMap

class LPFileObjClass(object):
    __slots__ = ('Sense','Name','VarToCoeff')

    def __init__(self):
        self.Sense = None
        self.Name = None
        self.VarToCoeff = {}

    def __call__(self):
        return self.Sense, self.Name, self.VarToCoeff

    def __getitem__(self, index):
        if index == 0:
            return self.Sense
        if index == 1:
            return self.Name
        if index == 2:
            return self.VarToCoeff

    # store the objective information from a LP file
    def AssignSense(self, SenseIn):
        if ((SenseIn[0] == 'min') or (SenseIn[0] == 'max')):
            print(("The obj sense in the lp file does not seem to be valid: "+str(SenseIn)))
            sys.exit(1)
        self.Sense = SenseIn

    def AssignName(self, arg):
        self.Name = arg

    def AddToMap(self, Coeff, Var):
        # check to see if Coeff is a number
        self.VarToCoeff[Var] = float(Coeff)
        #print "in OBJClass: self.VarToCoeff=", self.VarToCoeff

class LPFileConstraintClass(object):

    __slots__ = ('VarToCoeff','Comparator','RHS')

    def __init__(self):
        self.VarToCoeff = {}
        self.Comparator = ''
        self.RHS = None

    def __call__(self):
        return self.VarToCoeff, self.Comparator, self.RHS

    def __getitem__(self, index):
        if index == 0:
            return self.VarToCoeff
        if index == 1:
            return self.Comparator
        if index == 2:
            return self.RHS

    def AddToMap(self, Coeff, Var):
        # Check if Coeff is a comparator. Then we know if we reached
        # the end of the constraint.
        if Coeff in ('=', '<', '>', '<=', '>='):
            comparator = Coeff
            rhs = Var
            return "EndOfConstraint"
        else:
            self.VarToCoeff[Var] = float(Coeff)

    def AssignComparator(self, arg):
        self.Comparator = arg
        #print "self.Comparator=", self.Comparator

    def AssignRHS(self, arg):
        self.RHS = arg
        #print "self.RHS =",self.RHS
