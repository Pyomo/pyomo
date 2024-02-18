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

import random
import os.path
import pyomo.scripting.convert

random.seed(2384792387)

nsets = [i * 1000 for i in range(1, 8, 4)]
nelts = [i * 1000 for i in range(1, 61, 10)]
seeds = [random.getrandbits(32) for i in range(10)]

for seed in seeds:
    random.seed(seed)

    for m in nsets:
        for n in nelts:
            fname = 'scover_%d_%d_%d' % (n, m, seed)
            print('fname', fname)
            pyomo.scripting.convert.pyomo2lp(
                args=[
                    '--model-options',
                    'n=%d m=%d seed=%d type=fixed_element_coverage rho=0.1'
                    % (n, m, seed),
                    '--save-model',
                    '%s.lp' % fname,
                    os.path.abspath('../sc.py'),
                ]
            )
