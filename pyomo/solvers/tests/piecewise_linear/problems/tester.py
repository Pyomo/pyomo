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

import os

from pyomo.common.fileutils import import_file, this_file_dir
from pyomo.environ import Var, maximize, value
from pyomo.opt import SolverFactory

solver_name = 'cplexamp'
opt = SolverFactory(solver_name, solve_io='nl')

kwds = {'pw_constr_type': 'UB', 'pw_repn': 'DCC', 'sense': maximize, 'force_pw': True}

problem_names = []
problem_names.append("piecewise_multi_vararray")
problem_names.append("concave_multi_vararray1")
problem_names.append("concave_multi_vararray2")
problem_names.append("convex_multi_vararray1")
problem_names.append("convex_multi_vararray2")
problem_names.append("convex_vararray")
problem_names.append("concave_vararray")
problem_names.append("convex_var")
problem_names.append("concave_var")
problem_names.append("piecewise_var")
problem_names.append("piecewise_vararray")
problem_names.append("step_var")
problem_names.append("step_vararray")

problem_names = ['convex_var']

if __name__ == '__main__':
    for problem_name in problem_names:
        p = import_file(os.path.join(this_file_dir(), problem_name) + '.py')

        model = p.define_model(**kwds)
        inst = model.create_instance()

        results = opt.solve(inst, tee=True)

        res = dict()
        for block in inst.block_data_objects(active=True):
            for variable in block.component_map(Var, active=True).values():
                for var in variable.values():
                    name = var.name
                    if (name[:2] == 'Fx') or (name[:1] == 'x'):
                        res[name] = value(var)
        print(res)
