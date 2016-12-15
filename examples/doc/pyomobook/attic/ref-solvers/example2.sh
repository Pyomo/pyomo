#!/bin/sh

# @cmd:
pyomo solve --solver=cbc --solver-io=nl simple.py
# @:cmd
cat results.yml
