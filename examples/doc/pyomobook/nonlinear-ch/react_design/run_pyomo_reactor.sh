#!/bin/sh

# @cmd:
pyomo solve --solver=ipopt --summary --stream-solver ReactorDesign.py
# @:cmd
rm -f results.yml
