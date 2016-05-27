#!/bin/sh

cd ../overview
# @cmd:
pyomo convert --output=concrete1.lp concrete1.py
# @:cmd
diff concrete1.lp ../command/concrete1.lp
