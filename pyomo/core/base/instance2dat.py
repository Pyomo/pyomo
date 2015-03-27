#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

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
                print >>output_file, "set "+set_name+" := "
                for element in set_object:
                    print >>output_file, element
                print >>output_file, ";"
            elif set_object.dim() == 1:
                for index in set_object:
                    print >>output_file, "set "+set_name+"[\""+str(index)+"\"]"+" := ",
                    for element in set_object[index]:
                        print >>output_file, element,
                    print >>output_file, ";"
            else:
                print >>output_file, "***MULTIPLY INDEXED SETS NOT IMPLEMENTED!!!"
                pass

            print >>output_file, ""

    for param_name, param_object in iteritems(instance.component_map(Param, active=True)):
        if (param_object._initialize is not None) and (type(param_object._initialize) is types.FunctionType):
            continue
        elif len(param_object) == 0:
            continue

        if None in param_object:
            print >>output_file, "param "+param_name+" := "+str(value(param_object[None]))+" ;"
            print >>output_file, ""
        else:
            print >>output_file, "param "+param_name+" := "
            if param_object.dim() == 1:
                for index in param_object:
                    print >>output_file, index, str(value(param_object[index]))
            else:
                for index in param_object:
                    for i in index:
                        print >>output_file, i,
                    print >>output_file, str(value(param_object[index]))
            print >>output_file, ";"
            print >>output_file, ""

    output_file.close()
