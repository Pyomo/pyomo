#!/bin/sh

# @cmd:
pyomo convert --output=concrete1.nl concrete1.py
# @:cmd
diff concrete1.nl ../command-ch/concrete1-ref.nl
