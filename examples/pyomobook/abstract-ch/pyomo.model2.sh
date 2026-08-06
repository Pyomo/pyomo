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
pyomo convert --output=concrete1.lp concrete1.py
# @:cmd
diff concrete1.lp ../command-ch/concrete1-ref.lp
rm -f results.yml results.json concrete1.lp
