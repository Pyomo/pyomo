#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

__all__ = ['instance2dat']

import types
from six import iteritems
from pyomo.core.base import Set, Param, value


# IMPT: Only works on non-nested block models at the moment!

def instance2dat(instance, output_filename):

    output_file = open(output_filename,"w")

    for set_name, set_object in iteritems(instance.component_map(Set, active=True)):
        if (set_object.initialize is not None) and (type(set_object.initialize) is types.FunctionType):
            continue

        if (set_name.find("_index") == -1) and (set_name.find("_domain") == -1):
            if set_object.dim() == 0:
                if len(set_object) == 0:
                    continue
                output_file.write("set " + set_name + " := \n")
                for element in set_object:
                    output_file.write(element,)
                output_file.write(";\n")
            elif set_object.dim() == 1:
                for index in set_object:
                    output_file.write("set " + set_name + "[\""+str(index) + "\"]"+" :=")
                    for element in set_object[index]:
                        output_file.write(element,)
                    output_file.write(";\n")
            else:
                output_file.write("***MULTIPLY INDEXED SETS NOT IMPLEMENTED!!!\n")
                pass

            output_file.write("\n")

    for param_name, param_object in iteritems(instance.component_map(Param, active=True)):
        if (param_object._initialize is not None) and (type(param_object._initialize) is types.FunctionType):
            continue
        elif len(param_object) == 0:
            continue

        if None in param_object:
            output_file.write("param "+param_name+" := "
                              + str(value(param_object[None])) + " ;\n")
            output_file.write("\n")
        else:
            output_file.write("param " + param_name + " := \n")
            if param_object.dim() == 1:
                for index in param_object:
                    output_file.write(str(index) + str(value(param_object[index])) + "\n")
            else:
                for index in param_object:
                    for i in index:
                        output_file.write(i,)
                    output_file.write(str(value(param_object[index])) + "\n")
            output_file.write(";\n")
            output_file.write("\n")

    output_file.close()
