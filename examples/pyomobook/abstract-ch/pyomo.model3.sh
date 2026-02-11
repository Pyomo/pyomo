#!/bin/sh
# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

# @cmd:
pyomo convert --output=concrete1.nl concrete1.py
# @:cmd
python -m pyomo.repn.tests.nl_diff concrete1.nl concrete1-ref.nl
rm -f results.yml results.json concrete1.nl
