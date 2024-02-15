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


def verify(obj, x, iv):
    assert obj == 2
    for i in range(3):
        assert sorted([iv[i, 0], iv[i, 1]]) == [0, 1]
        assert iv[i, 0] == (0 if x[i] else 1)
        assert iv[i, 1] == (1 if x[i] else 0)
        assert sum(x.values()) >= 7
    fname = os.path.basename(__file__)
    if fname.endswith('.pyc'):
        fname = fname[:-1]
    print("%s: OK: result validated" % (fname,))


def verify_file(fname):
    import yaml

    ans = yaml.load(open(fname, 'r'), Loader=yaml.FullLoader)
    assert ans['Solution'][0]['number of solutions'] == 1
    obj = ans['Solution'][1]['Objective']['o']['Value']
    ZERO = {'Value': 0}
    x = {}
    iv = {}
    for i in range(3):
        x[i] = ans['Solution'][1]['Variable'].get('x[%s]' % i, ZERO)['Value']
        iv[i, 0] = ans['Solution'][1]['Variable'].get(
            'd_0[%s].binary_indicator_var' % i, ZERO
        )['Value']
        iv[i, 1] = ans['Solution'][1]['Variable'].get(
            'd_nonzero[%s].binary_indicator_var' % i, ZERO
        )['Value']
    verify(obj, x, iv)


def verify_model(model):
    assert len(model.solutions) == 1
    obj = model.o()
    x = {}
    iv = {}
    for i in range(3):
        x[i] = model.x[i]()
        iv[i, 0] = model.d_0[i].binary_indicator_var()
        iv[i, 1] = model.d_nonzero[i].binary_indicator_var()
    verify(obj, x, iv)


if __name__ == '__main__':
    import sys

    verify_file(sys.argv[1])
