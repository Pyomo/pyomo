#!/bin/sh

cd ../overview
# @cmd:
pyomo convert --output=concrete1.nl concrete1.py
# @:cmd
diff concrete1.nl ../command/concrete1.nl
