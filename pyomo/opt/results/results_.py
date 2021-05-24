#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

__all__ = ['SolverResults']

import math
import sys
import copy
import json
import logging
import os.path

from pyomo.common.dependencies import yaml, yaml_load_args, yaml_available
import pyomo.opt
from pyomo.opt.results.container import (undefined,
                                         ignore,
                                         ListContainer,
                                         MapContainer)
import pyomo.opt.results.solution
from pyomo.opt.results.solution import default_print_options as dpo
import pyomo.opt.results.problem
import pyomo.opt.results.solver

from io import StringIO

logger = logging.getLogger(__name__)

class SolverResults(MapContainer):

    undefined = undefined
    default_print_options = dpo

    def __init__(self):
        MapContainer.__init__(self)
        self._sections = []
        self._descriptions = {}
        self.add('problem',
                 ListContainer(pyomo.opt.results.problem.ProblemInformation),
                 False,
                 "Problem Information")
        self.add('solver',
                 ListContainer(pyomo.opt.results.solver.SolverInformation),
                 False,
                 "Solver Information")
        self.add('solution',
                 pyomo.opt.results.solution.SolutionSet(),
                 False,
                 "Solution Information")

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
        _fmt = kwds.pop('format', None)
        if _fmt:
            _fmt = _fmt.lower()
        fname = kwds.pop('filename', None)

        if fname:
            ext = os.path.splitext(fname)[1].lstrip('.')
            normalized_ext = {
                'json': 'json',
                'jsn': 'json',
                'yaml': 'yaml',
                'yml': 'yaml',
            }.get(ext, None)
            if not _fmt:
                _fmt = normalized_ext
            elif normalized_ext and _fmt != normalized_ext:
                logger.warning(
                    "writing results to file (%s) using what appears "
                    "to be an incompatible format (%s)" % (fname, _fmt))
            with open(fname, "w") as OUTPUT:
                kwds['ostream'] = OUTPUT
                kwds['format'] = _fmt
                self.write(**kwds)
        else:
            if not _fmt:
                _fmt = 'yaml'
            if _fmt == 'yaml':
                self.write_yaml(**kwds)
            elif _fmt == 'json':
                self.write_json(**kwds)
            else:
                raise ValueError("Unknown results file format: %s" % (_fmt,))

    def write_json(self, **kwds):
        if 'ostream' in kwds:
            ostream = kwds['ostream']
            del kwds['ostream']
        else:
            ostream = sys.stdout

        option = copy.copy(SolverResults.default_print_options)
        # TODO: verify that we need this for-loop
        for key in kwds:
            setattr(option,key,kwds[key])
        repn = self.json_repn(option)

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
                    soln[data] = "No values"
                    continue
                for kk,vv in data_value.items():
                    # TODO: remove this if-block.  This is a hack
                    if not type(vv) is dict:
                        vv = {'Value':vv}
                    tmp = {}
                    for k,v in vv.items():
                        # TODO: remove this if-block.  This is a hack
                        if v is not None and math.fabs(v) > 1e-16:
                            tmp[k] = v
                    if len(tmp) > 0:
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
        # TODO: verify that we need this for-loop
        for key in kwds:
            setattr(option,key,kwds[key])
        repn = self._repn_(option)

        ostream.write("# ==========================================================\n")
        ostream.write("# = Solver Results                                         =\n")
        ostream.write("# ==========================================================\n")
        for i in range(len(self._order)):
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
            repn = yaml.load(istream, **yaml_load_args)
        else:
            repn = json.load(istream)
        for i in range(len(self._order)):
            key = self._order[i]
            if not key in repn:
                continue
            item = dict.__getitem__(self,key)
            item.load(repn[key])

    def __repr__(self):
        return str(self._repn_(SolverResults.default_print_options))

    def __str__(self):
        ostream = StringIO()
        option=SolverResults.default_print_options
        self.pprint(ostream, option, repn=self._repn_(option))
        return ostream.getvalue()


if __name__ == '__main__':
    results = SolverResults()
    results.write(schema=True)
    #print results
