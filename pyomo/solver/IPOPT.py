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
        version = version.split(' ')[1]
        version = version.strip()
        version = tuple(int(i) for i in version.split('.'))
        return version

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, val):
        self._config = val

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
                info = nl_writer.write(
                    model,
                    nl_file,
                    row_file,
                    col_file,
                    symbolic_solver_labels=config.symbolic_solver_labels,
                )
            # Call IPOPT - passing the files via the subprocess
            subprocess.run()
