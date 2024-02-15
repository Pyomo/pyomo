#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2024
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

# abstract7.py
import pyomo.environ as pyo
import pickle
import sys
from os.path import dirname, abspath


# @preprocess:
def pyomo_preprocess(options=None):
    print("Here are the options that were provided:")
    if options is not None:
        options.display()


# @:preprocess


# @create_model:
def pyomo_create_model(options=None, model_options=None):
    sys.path.append(abspath(dirname(__file__)))
    abstract6 = __import__('abstract6')
    sys.path.remove(abspath(dirname(__file__)))
    return abstract6.Model


# @:create_model


# @create_modeldata:
def pyomo_create_dataportal(options=None, model=None):
    data = pyo.DataPortal(model=model)
    data.load(filename='abstract6.dat')
    return data


# @:create_modeldata


# @print_model:
def pyomo_print_model(options=None, model=None):
    if options['runtime']['logging']:
        model.pprint()


# @:print_model


# @modify_instance:
def pyomo_modify_instance(options=None, model=None, instance=None):
    instance.x[1].value = 0.0
    instance.x[1].fixed = True


# @:modify_instance


# @print_instance:
def pyomo_print_instance(options=None, instance=None):
    if options['runtime']['logging']:
        instance.pprint()


# @:print_instance


# @save_instance:
def pyomo_save_instance(options=None, instance=None):
    OUTPUT = open('abstract7.pyomo', 'w')
    OUTPUT.write(str(pickle.dumps(instance)))
    OUTPUT.close()


# @:save_instance


# @print_results:
def pyomo_print_results(options=None, instance=None, results=None):
    print(results)


# @:print_results


# @save_results:
def pyomo_save_results(options=None, instance=None, results=None):
    OUTPUT = open('abstract7.results', 'w')
    OUTPUT.write(str(results))
    OUTPUT.close()


# @:save_results


# @postprocess:
def pyomo_postprocess(options=None, instance=None, results=None):
    instance.solutions.load_from(results, allow_consistent_values_for_fixed_vars=True)
    print("Solution value " + str(pyo.value(instance.obj)))


# @:postprocess
