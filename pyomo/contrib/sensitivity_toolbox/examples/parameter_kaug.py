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
#
# The parametric.mod example from sIPOPT
#
# Original implementation by Hans Pirnay is in pyomo/examples/pyomo/suffixes
#

from pyomo.environ import *
from pyomo.contrib.sensitivity_toolbox.sens import sensitivity_calculation


def create_model():
    '''Create a concrete Pyomo model for this example'''
    m = ConcreteModel()

    m.x1 = Var(initialize=0.15, within=NonNegativeReals)
    m.x2 = Var(initialize=0.15, within=NonNegativeReals)
    m.x3 = Var(initialize=0.0, within=NonNegativeReals)

    m.eta1 = Param(initialize=4.5, mutable=True)
    m.eta2 = Param(initialize=1.0, mutable=True)

    m.const1 = Constraint(expr=6 * m.x1 + 3 * m.x2 + 2 * m.x3 - m.eta1 == 0)
    m.const2 = Constraint(expr=m.eta2 * m.x1 + m.x2 - m.x3 - 1 == 0)
    m.cost = Objective(expr=m.x1**2 + m.x2**2 + m.x3**2)

    return m


def run_example(print_flag=True):
    '''
    Execute the example

    Arguments:
        print_flag: Toggle on/off printing

    Returns
        sln_dict: Dictionary containing solution (used for automated testing)

    '''
    m = create_model()

    m.perturbed_eta1 = Param(initialize=4.0)
    m.perturbed_eta2 = Param(initialize=1.0)

    m_kaug_dsdp = sensitivity_calculation(
        'k_aug', m, [m.eta1, m.eta2], [m.perturbed_eta1, m.perturbed_eta2], tee=True
    )

    if print_flag:
        print("\nOriginal parameter values:")
        print("\teta1 =", m.eta1())
        print("\teta2 =", m.eta2())

        print("Initial point:")
        print("\tObjective =", value(m.cost))
        print("\tx1 =", m.x1())
        print("\tx2 =", m.x2())
        print("\tx3 =", m.x3())

        # Kaug saves only approximated solutions not original solutions
        print("\nNew parameter values:")
        print("\teta1 =", m_kaug_dsdp.perturbed_eta1())
        print("\teta2 =", m_kaug_dsdp.perturbed_eta2())

        print("(Approximate) solution with the new parameter values:")
        print("\tObjective =", m_kaug_dsdp.cost())
        print("\tx1 =", m_kaug_dsdp.x1())
        print("\tx2 =", m_kaug_dsdp.x2())
        print("\tx3 =", m_kaug_dsdp.x3())

    # Save the results in a dictionary.
    # This is optional and makes automated testing convenient.
    # This code is not required for a Minimum Working Example (MWE)
    d = dict()
    d['eta1'] = m.eta1()
    d['eta2'] = m.eta2()
    d['x1_init'] = m.x1()
    d['x2_init'] = m.x2()
    d['x3_init'] = m.x3()
    d['eta1_pert'] = m_kaug_dsdp.perturbed_eta1()
    d['eta2_pert'] = m_kaug_dsdp.perturbed_eta2()
    d['cost_pert'] = m_kaug_dsdp.cost()
    d['x1_pert'] = m_kaug_dsdp.x1()
    d['x2_pert'] = m_kaug_dsdp.x2()
    d['x3_pert'] = m_kaug_dsdp.x3()

    return d


if __name__ == '__main__':
    d = run_example()
