#  _________________________________________________________________________
#
#  Pyomo: A COmmon Optimization Python Repository
#  Copyright (c) 2008 Sandia Corporation.
#  This software is distributed under the BSD License.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  For more information, see the Pyomo README.txt file.
#  _________________________________________________________________________

__all__ = ['SolverResults']

import math
import sys
import copy
import json
from six import iteritems
from six.moves import xrange
try:
    import StringIO
except ImportError:
    import io as StringIO
try:
    import yaml
    yaml_available=True
except ImportError:
    yaml_available=False

from pyutilib.enum import Enum
from pyutilib.misc import Bunch

from pyomo.opt.results.container import *
from pyomo.opt.results.solution import default_print_options as dpo
from pyomo.opt.results.problem import ProblemInformation
from pyomo.opt.results.solver import SolverInformation
from pyomo.opt.results.solution import SolutionSet


class SolverResults(MapContainer):

    undefined = undefined
    default_print_options = dpo

    def X__del__(self):
        MapContainer.__del__(self)
        self._sections = None
        self._descriptions = None
        self._symbol_map = None

    def __init__(self):
        MapContainer.__init__(self)
        self._sections = []
        self._symbol_map = None
        self._descriptions = {}
        self.add('problem', ListContainer(ProblemInformation), False, "Problem Information")
        self.add('solver', ListContainer(SolverInformation), False, "Solver Information")
        self.add('solution', SolutionSet(), False, "Solution Information")

    def __getstate__(self):
        def _canonical_label(obj):
            if obj is obj.parent_component():
                label = obj.name
            else:
                index = obj.index
                if type(index) is not tuple:
                    index = (index,)
                codedIdx = []
                for idx in index:
                    if idx is None:
                        codedIdx.append('!')
                    elif type(idx) is str:
                        codedIdx.append('$'+idx)
                    elif int(idx) == idx:
                        codedIdx.append('#'+str(idx))
                    else:
                        raise ValueError(
                            "Unexpected type %s encountered when pickling "
                            "SolverResults object index: %s" %
                            (str(type(idx)), str(obj.index)))
                obj = obj.parent_component()
                label = obj.name + ':' + ','.join(codedIdx)
            if obj._parent is None or obj._parent() is None:
                return label
            obj = obj._parent()
            while obj._parent is not None and obj._parent() is not None:
                label = str(obj.name) + '.' + label
                obj = obj._parent()
            return label
            
        sMap = self._symbol_map
        if sMap is None:
            return MapContainer.__getstate__(self)
        for soln in self.solution:
            for symbol, obj in iteritems(soln.objective):
                obj.canonical_label = _canonical_label(sMap.getObject(symbol))
            for symbol, var in iteritems(soln.variable):
                if symbol == 'ONE_VAR_CONSTANT':
                    continue
                var['canonical_label'] = _canonical_label(sMap.getObject(symbol))
            for symbol, con in iteritems(soln.constraint):
                obj_ = sMap.getObject(symbol)
                if obj_ is not sMap.UnknownSymbol:
                    con["canonical_label"] = _canonical_label(sMap.getObject(symbol))
        results = MapContainer.__getstate__(self)
        results['_symbol_map'] = None
        return results

    def add(self, name, value, active, description):
        self.declare(name, value=value, active=active)
        tmp = self._convert(name)
        self._sections.append(tmp)
        self._descriptions[tmp]=description

    def json_repn(self, options=None):
        if options is None:
            return self._repn_(SolverResults.default_print_options)
        else:
            return self._repn_(options)

    def _repn_(self, option):
        if not option.schema and not self._active and not self._required:
            return ignore
        tmp = {}
        for key in self._sections:
            rep = dict.__getitem__(self, key)._repn_(option)
            if not rep == ignore:
                tmp[key] = rep
        return tmp

    def write(self, **kwds):
        if 'filename' in kwds:
            OUTPUT=open(kwds['filename'],"w")
            del kwds['filename']
            kwds['ostream']=OUTPUT
            self.write(**kwds)
            OUTPUT.close()
            return

        if not 'format' in kwds or kwds['format'] == 'yaml':
            self.write_yaml(**kwds)
            return
        #
        # Else, write in JSON format
        #
        repn = self.json_repn()
        if 'ostream' in kwds:
            ostream = kwds['ostream']
            del kwds['ostream']
        else:
            ostream = sys.stdout
        
        for soln in repn.get('Solution', []):
            for data in ['Variable', 'Constraint', 'Objective']:
                remove = set()
                if data not in soln:
                    continue
                data_value = soln[data]
                if not isinstance(data_value,dict):
                    continue
                if not data_value:
                    # a variable/constraint/objective may have no
                    # entries, e.g., if duals or slacks weren't
                    # extracted in a solution.
                    # FIXME: technically, the "No nonzero values" message is 
                    #       incorrect - it could simply be "No values". this
                    #       would unfortunatley require updating of a ton of
                    #       test baselines.
                    soln[data] = "No nonzero values"
                    continue
                for kk,vv in iteritems(data_value):
                    tmp = {}
                    for k,v in iteritems(vv):
                        if k == 'Id' or ( v is not None and math.fabs(v) ):
                            tmp[k] = v
                    if len(tmp) > 1 or ( tmp and 'Id' not in tmp ):
                        soln[data][kk] = tmp
                    else:
                        remove.add((data,kk))
                for item in remove:
                    del soln[item[0]][item[1]]
        json.dump(repn, ostream, indent=4, sort_keys=True)

    def write_yaml(self, **kwds):
        if 'ostream' in kwds:
            ostream = kwds['ostream']
            del kwds['ostream']
        else:
            ostream = sys.stdout

        option = copy.copy(SolverResults.default_print_options)
        for key in kwds:
            setattr(option,key,kwds[key])

        repn = self._repn_(option)
        ostream.write("# ==========================================================\n")
        ostream.write("# = Solver Results                                         =\n")
        ostream.write("# ==========================================================\n")
        for i in xrange(len(self._order)):
            key = self._order[i]
            if not key in repn:
                continue
            item = dict.__getitem__(self,key)
            ostream.write("# ----------------------------------------------------------\n")
            ostream.write("#   %s\n" % self._descriptions[key])
            ostream.write("# ----------------------------------------------------------\n")
            ostream.write(key+": ")
            if isinstance(item, ListContainer):
                item.pprint(ostream, option, prefix="", repn=repn[key])
            else:
                item.pprint(ostream, option, prefix="  ", repn=repn[key])

    def read(self, **kwds):
        if 'istream' in kwds:
            istream = kwds['istream']
            del kwds['istream']
        else:
            ostream = sys.stdin
        if 'filename' in kwds:
            INPUT=open(kwds['filename'],"r")
            del kwds['filename']
            kwds['istream']=INPUT
            self.read(**kwds)
            INPUT.close()
            return

        if not 'format' in kwds or kwds['format'] == 'yaml':
            if not yaml_available:
                raise IOError("Aborting SolverResults.read() because PyYAML is not installed!")
            repn = yaml.load(istream, Loader=yaml.SafeLoader)
        else:
            repn = json.load(istream)
        for i in xrange(len(self._order)):
            key = self._order[i]
            if not key in repn:
                continue
            item = dict.__getitem__(self,key)
            item.load(repn[key])

    def __repr__(self):
        return str(self._repn_(SolverResults.default_print_options))

    def __str__(self):
        ostream = StringIO.StringIO()
        option=SolverResults.default_print_options
        self.pprint(ostream, option, repn=self._repn_(option))
        return ostream.getvalue()


if __name__ == '__main__':
    results = SolverResults()
    results.write(schema=True)
    #print results
