#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2022
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import os
import subprocess

from pyomo.common import Executable
from pyomo.common.config import ConfigValue
from pyomo.common.tempfiles import TempfileManager
from pyomo.opt import WriterFactory
from pyomo.solver.base import SolverBase
from pyomo.solver.config import SolverConfig
from pyomo.solver.factory import SolverFactory
from pyomo.solver.results import Results, TerminationCondition, SolutionStatus
from pyomo.solver.solution import SolutionLoaderBase
from pyomo.solver.util import SolverSystemError

import logging

logger = logging.getLogger(__name__)


class IPOPTConfig(SolverConfig):
    def __init__(
        self,
        description=None,
        doc=None,
        implicit=False,
        implicit_domain=None,
        visibility=0,
    ):
        super().__init__(
            description=description,
            doc=doc,
            implicit=implicit,
            implicit_domain=implicit_domain,
            visibility=visibility,
        )

        self.executable = self.declare(
            'executable', ConfigValue(default=Executable('ipopt'))
        )
        self.save_solver_io: bool = self.declare(
            'save_solver_io', ConfigValue(domain=bool, default=False)
        )


class IPOPTSolutionLoader(SolutionLoaderBase):
    pass


@SolverFactory.register('ipopt', doc='The IPOPT NLP solver (new interface)')
class IPOPT(SolverBase):
    CONFIG = IPOPTConfig()

    def __init__(self, **kwds):
        self.config = self.CONFIG(kwds)

    def available(self):
        if self.config.executable.path() is None:
            return self.Availability.NotFound
        return self.Availability.FullLicense

    def version(self):
        results = subprocess.run(
            [str(self.config.executable), '--version'],
            timeout=1,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        )
        version = results.stdout.splitlines()[0]
        version = version.split(' ')[1].strip()
        version = tuple(int(i) for i in version.split('.'))
        return version

    @property
    def config(self):
        return self.config

    @config.setter
    def config(self, val):
        self.config = val

    def solve(self, model, **kwds):
        # Check if solver is available
        avail = self.available()
        if not avail:
            raise SolverSystemError(
                f'Solver {self.__class__} is not available ({avail}).'
            )
        # Update configuration options, based on keywords passed to solve
        config = self.config(kwds.pop('options', {}))
        config.set_value(kwds)
        # Get a copy of the environment to pass to the subprocess
        env = os.environ.copy()
        if 'PYOMO_AMPLFUNC' in env:
            env['AMPLFUNC'] = "\n".join(
                filter(
                    None, (env.get('AMPLFUNC', None), env.get('PYOMO_AMPLFUNC', None))
                )
            )
        # Write the model to an nl file
        nl_writer = WriterFactory('nl')
        # Need to add check for symbolic_solver_labels; may need to generate up
        # to three files for nl, row, col, if ssl == True
        # What we have here may or may not work with IPOPT; will find out when
        # we try to run it.
        with TempfileManager.new_context() as tempfile:
            dname = tempfile.mkdtemp()
            with open(os.path.join(dname, model.name + '.nl')) as nl_file, open(
                os.path.join(dname, model.name + '.row')
            ) as row_file, open(os.path.join(dname, model.name + '.col')) as col_file:
                self.info = nl_writer.write(
                    model,
                    nl_file,
                    row_file,
                    col_file,
                    symbolic_solver_labels=config.symbolic_solver_labels,
                )
            # Call IPOPT - passing the files via the subprocess
            cmd = [str(config.executable), nl_file, '-AMPL']
            if config.time_limit is not None:
                config.solver_options['max_cpu_time'] = config.time_limit
            for key, val in config.solver_options.items():
                cmd.append(key + '=' + val)
            process = subprocess.run(
                cmd, timeout=config.time_limit, env=env, universal_newlines=True
            )

            if process.returncode != 0:
                if self.config.load_solution:
                    raise RuntimeError(
                        'A feasible solution was not found, so no solution can be loaded.'
                        'Please set config.load_solution=False and check '
                        'results.termination_condition and '
                        'results.incumbent_objective before loading a solution.'
                    )
                results = Results()
                results.termination_condition = TerminationCondition.error
            else:
                results = self._parse_solution()

    def _parse_solution(self):
        # STOPPING POINT: The suggestion here is to look at the original
        # parser, which hasn't failed yet, and rework it to be ... better?
        pass
