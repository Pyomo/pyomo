# demontrations/tests of/for daps classes - DLW Dec 2016
# NOTE: some file and directory references may need to be edited.
# NOTE: If you want to run these as serious tests, you might
#       want to delete a few files before you run.
import os
import pyomo.pysp.plugins.csvsolutionwriter as csvw
import basicclasses as bc
import distr2pysp as dp
import stoch_solver as st

# concrete as needed by IDAES (serial):
print ("\n*** 2Stage_json")
tree_model = bc.Tree_2Stage_json_dir('concrete_farmer', 'TreeTemplateFile.json')

stsolver = st.StochSolver('cref.py', tree_model)
stsolver.solve_ef('cplex')
# the stsolver.scenario_tree has the solution
csvw.write_csv_soln(stsolver.scenario_tree, "testcref")
kadsjhf
### simple tests ####
print("\n*** 2Stage_AMPL")
bc.do_2Stage_AMPL_dir('farmer', 'TreeTemplateFile.dat', 'ScenTemplate.dat')
os.system('runef -m farmer/ReferenceModel.py -s farmer')

print("\n*** 2Stage_AMPL again")
bc.do_2Stage_AMPL_dir('farmer', 'TreeTemplateFile.dat', \
                      'scen_template_with_tokens.dat', \
                      '#STARTSCEN', '#ENDSCEN')
os.system('runef -m farmer/ReferenceModel.py -s farmer')


#==== distrs ====
print("\n*** indep norms")
dp.indep_norms_from_data_2stage('concrete_farmer/dptest/datafiledict.json', 
                                'concrete_farmer/TreeTemplateFile.dat',
                                4,
                                'concrete_farmer/dptest',
                                Seed = 7734)
os.system('runef -m concrete_farmer/dptest/ReferenceModel.py -s concrete_farmer/dptest')

print ("\n*** scipy")
dp.json_scipy_2stage('concrete_farmer/dptest/distrdict.json', 
                     'concrete_farmer/TreeTemplateFile.dat',
                     3,
                     'concrete_farmer/dptest')
os.system('runef -m concrete_farmer/dptest/ReferenceModel.py -s concrete_farmer/dptest')
