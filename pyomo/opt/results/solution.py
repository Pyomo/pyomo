#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

__all__ = ['SolutionStatus', 'Solution']

import math
from six import iterkeys, iteritems
from six.moves import xrange
import enum
from pyomo.opt.results.container import MapContainer, ListContainer, ignore
from pyomo.common.collections import Bunch, OrderedDict

default_print_options = Bunch(schema=False,
                              sparse=True,
                              num_solutions=None,
                              ignore_time=False,
                              ignore_defaults=False)

class SolutionStatus(str, enum.Enum):
    bestSoFar='bestSoFar'
    error='error'
    feasible='feasible'
    globallyOptimal='globallyOptimal'
    infeasible='infeasible'
    locallyOptimal='locallyOptimal'
    optimal='optimal'
    other='other'
    stoppedByLimit='stoppedByLimit'
    unbounded='unbounded'
    unknown='unknown'
    unsure='unsure'

    # Overloading __str__ is needed to match the behavior of the old
    # pyutilib.enum class (removed June 2020). There are spots in the
    # code base that expect the string representation for items in the
    # enum to not include the class name. New uses of enum shouldn't
    # need to do this.
    def __str__(self):
        return self.value


try:
    unicode
except NameError:
    basestring = unicode = str

try:
    long
    intlist = (int, long)
    numlist = (float, int, long)
except:
    intlist = (int, )
    numlist = (float, int)


class Solution(MapContainer):

    def __init__(self):
        MapContainer.__init__(self)

        self.declare('gap')
        self.declare('status', value=SolutionStatus.unknown)
        self.declare('message')

        self.declare('problem', value={})
        self.declare('objective', value={})
        self.declare('variable', value={})
        self.declare('constraint', value={})

        self._option = default_print_options

    def load(self, repn):
        # delete key from dictionary, call base class load, handle variable loading.
        if "Variable" in repn:
            tmp_ = repn["Variable"]
            del repn["Variable"]
            self.variable = tmp_
        if "Constraint" in repn:
            tmp_ = repn["Constraint"]
            del repn["Constraint"]
            self.constraint = tmp_
        if "Problem" in repn:
            tmp_ = repn["Problem"]
            del repn["Problem"]
            self.problem = tmp_
        if "Objective" in repn:
            tmp_ = repn["Objective"]
            del repn["Objective"]
            self.objective = tmp_
        MapContainer.load(self, repn)

    def pprint(self, ostream, option, from_list=False, prefix="", repn=None):
        #
        # the following is specialized logic for handling variable and
        # constraint maps - which are dictionaries of dictionaries, with
        # at a minimum an "id" element per sub-directionary.  
        #
        first = True
        for key in self._order:
            if not key in repn or key == 'Problem':
                continue
            item = dict.__getitem__(self,key)
            if not type(item.value) is dict:
                #
                # Do a normal print
                #
                if first:
                    ostream.write(key+": ")
                    first = False
                else:
                    ostream.write(prefix+key+": ")
                item.pprint(ostream, option, prefix=prefix+"  ", repn=repn[key])
            elif len(item.value) == 0:
                #
                # The dictionary is empty
                #
                ostream.write(prefix+key+": No values\n")
            else:
                print_zeros = key in ['Objective']
                first = True
                ostream.write(prefix+key+":")
                prefix_ = prefix
                prefix = prefix + "  "
                #
                # Print values in the dictionary
                #
                value = item.value
                id_ctr = 0
                id_dict_map = {}
                id_name_map = {} # the name could be an integer or float - so convert prior to       printing (see code below)
                id_nonzeros_map = {} # are any of the non-id entries are non-zero?
                entries_to_print = False

                for entry_name, entry_dict in iteritems(value):
                    entry_id = id_ctr
                    id_ctr += 1
                    id_name_map[entry_id] = entry_name
                    id_dict_map[entry_id] = entry_dict
                    id_nonzeros_map[entry_id] = False # until proven otherwise
                    for attr_name, attr_value in iteritems(entry_dict):
                        if print_zeros or math.fabs(attr_value) > 1e-16:
                            id_nonzeros_map[entry_id] = True
                            entries_to_print = True

                if entries_to_print:
                    for entry_id in sorted(iterkeys(id_dict_map), key=lambda id:id_name_map[id]):
                        if id_nonzeros_map[entry_id]:
                            if first:
                                ostream.write("\n")
                                first = False
                            ostream.write(prefix+str(id_name_map[entry_id])+":\n")
                            entry_dict = id_dict_map[entry_id]
                            for attr_name in sorted(iterkeys(entry_dict)):
                                attr_value = entry_dict[attr_name]
                                if isinstance(attr_value,float) and (math.floor(attr_value) == attr_value):
                                    attr_value = int(attr_value)
                                ostream.write(prefix+"  "+attr_name.capitalize()+": "+str(attr_value)+'\n')
                else:
                    ostream.write(" No nonzero values\n")
                prefix = prefix_



class SolutionSet(ListContainer):

    def __init__(self):
        ListContainer.__init__(self,Solution)
        self._option = default_print_options

    def _repn_(self, option):
        if not option.schema and not self._active and not self._required:
            return ignore
        if option.schema and len(self) == 0:
            self.add()
            self.add()
        if option.num_solutions is None:
            num = len(self)
        else:
            num = min(option.num_solutions, len(self))
        i=0
        tmp = []
        for item in self._list:
            tmp.append( item._repn_(option) )
            i=i+1
            if i == num:
                break
        return [OrderedDict([('number of solutions',len(self)), ('number of solutions displayed',num)])]+ tmp

    def __len__(self):
        return len(self._list)

    def __call__(self, i=1):
        return self._list[i-1]

    def pprint(self, ostream, option, prefix="", repn=None):
        if not option.schema and not self._active and not self._required:
            return ignore
        ostream.write("\n")
        ostream.write(prefix+"- ")
        spaces=""
        for key in repn[0]:
            ostream.write(prefix+spaces+key+": "+str(repn[0][key])+'\n')
            spaces="  "
        i=0
        for i in xrange(len(self._list)):
            item = self._list[i]
            ostream.write(prefix+'- ')
            item.pprint(ostream, option, from_list=True, prefix=prefix+"  ", repn=repn[i+1])

    def load(self, repn):
        #
        # Note: we ignore the first element of the repn list, since
        # it was generated on the fly by the SolutionSet object.
        #
        for data in repn[1:]: # repn items 1 through N are individual solutions.
            item = self.add()
            item.load(data)
