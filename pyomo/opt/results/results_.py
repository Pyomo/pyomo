#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import math
import sys
import copy
import json
import logging
import os.path

from pyomo.common.dependencies import yaml, yaml_load_args
import pyomo.opt
from pyomo.opt.results.container import undefined, ignore, ListContainer, MapContainer
import pyomo.opt.results.solution
from pyomo.opt.results.solution import default_print_options as dpo
import pyomo.opt.results.problem
import pyomo.opt.results.solver

from io import StringIO

logger = logging.getLogger(__name__)


def _guess_format(filename):
    "Return a standardized file format by looking at the filename extension"
    return {'.json': 'json', '.jsn': 'json', '.yaml': 'yaml', '.yml': 'yaml'}.get(
        os.path.splitext(filename)[1].lower(), None
    )


class SolverResults(MapContainer):
    undefined = undefined
    default_print_options = dpo

    def __init__(self):
        super().__init__()
        self._sections = []
        self._descriptions = {}
        self.add(
            'problem',
            ListContainer(pyomo.opt.results.problem.ProblemInformation),
            False,
            "Problem Information",
        )
        self.add(
            'solver',
            ListContainer(pyomo.opt.results.solver.SolverInformation),
            False,
            "Solver Information",
        )
        self.add(
            'solution',
            pyomo.opt.results.solution.SolutionSet(),
            False,
            "Solution Information",
        )

    def add(self, name, value, active, description):
        self.declare(name, value=value, active=active)
        tmp = self._convert(name)
        self._sections.append(tmp)
        self._descriptions[tmp] = description

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
        _format = kwds.pop('format', None)
        if _format:
            _format = _format.lower()
        if 'filename' in kwds:
            filename = kwds.pop('filename')
            _guess = _guess_format(filename)
            if not _format:
                _format = _guess
            if _guess and _format != _guess:
                logger.warning(
                    "writing results to file (%s) using what appears "
                    "to be an incompatible format (%s)" % (filename, _format)
                )
            if 'ostream' in kwds:
                raise ValueError("Cannot specify both filename and ostream")
            with open(filename, "w") as OUTPUT:
                return self.write(ostream=OUTPUT, format=_format, **kwds)

        ostream = kwds.pop('ostream', sys.stdout)
        if not _format:
            _format = 'yaml'
        if _format == 'yaml':
            self.write_yaml(ostream=ostream, **kwds)
        elif _format == 'json':
            self.write_json(ostream=ostream, **kwds)
        else:
            raise ValueError("Unknown results file format: %s" % (_format,))

    def write_json(self, **kwds):
        ostream = kwds.pop('ostream', sys.stdout)
        option = copy.copy(SolverResults.default_print_options)
        # TODO: verify that we need this for-loop
        for key, val in kwds.items():
            setattr(option, key, val)
        repn = self.json_repn(option)

        for soln in repn.get('Solution', []):
            for data in ['Variable', 'Constraint', 'Objective']:
                remove = set()
                if data not in soln:
                    continue
                data_value = soln[data]
                if not isinstance(data_value, dict):
                    continue
                if not data_value:
                    # a variable/constraint/objective may have no
                    # entries, e.g., if duals or slacks weren't
                    # extracted in a solution.
                    soln[data] = "No values"
                    continue
                for kk, vv in data_value.items():
                    # TODO: remove this if-block.  This is a hack
                    if not type(vv) is dict:
                        vv = {'Value': vv}
                    tmp = {}
                    for k, v in vv.items():
                        # TODO: remove this if-block.  This is a hack
                        if v is not None and math.fabs(v) > 1e-16:
                            tmp[k] = v
                    if len(tmp) > 0:
                        soln[data][kk] = tmp
                    else:
                        remove.add((data, kk))
                for item in remove:
                    del soln[item[0]][item[1]]
        json.dump(repn, ostream, indent=4, sort_keys=True)

    def write_yaml(self, **kwds):
        ostream = kwds.pop('ostream', sys.stdout)
        option = copy.copy(SolverResults.default_print_options)
        # TODO: verify that we need this for-loop
        for key in kwds:
            setattr(option, key, kwds[key])
        repn = self._repn_(option)

        ostream.write("# ==========================================================\n")
        ostream.write("# = Solver Results                                         =\n")
        ostream.write("# ==========================================================\n")
        for key, item in self.items():
            if not key in repn:
                continue
            ostream.write(
                "# ----------------------------------------------------------\n"
            )
            ostream.write("#   %s\n" % self._descriptions[key])
            ostream.write(
                "# ----------------------------------------------------------\n"
            )
            ostream.write(key + ": ")
            if isinstance(item, ListContainer):
                item.pprint(ostream, option, prefix="", repn=repn[key])
            else:
                item.pprint(ostream, option, prefix="  ", repn=repn[key])

    def read(self, **kwds):
        _format = kwds.pop('format', None)
        if _format:
            _format = _format.lower()
        if 'filename' in kwds:
            filename = kwds.pop('filename')
            if not _format:
                _format = _guess_format(filename)
            if 'istream' in kwds:
                raise ValueError("Cannot specify both filename and istream")
            with open(filename, "r") as INPUT:
                return self.read(istream=INPUT, format=_format, **kwds)

        istream = kwds.pop('istream', sys.stdin)
        if not _format or _format == 'yaml':
            repn = yaml.load(istream, **yaml_load_args)
        elif _format == 'json':
            repn = json.load(istream)
        else:
            raise ValueError(f"Unknown SolverResults format: '{_format}'")
        for key, item in repn.items():
            dict.__getitem__(self, key).load(item)

    def __repr__(self):
        return str(self._repn_(SolverResults.default_print_options))

    def __str__(self):
        ostream = StringIO()
        option = SolverResults.default_print_options
        self.pprint(ostream, option, repn=self._repn_(option))
        return ostream.getvalue()


if __name__ == '__main__':
    results = SolverResults()
    results.write(schema=True)
    # print results
