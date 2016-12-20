#!/bin/sh

# @cmd:
pyomo solve --solver=cbc --solver-manager=neos simple.py
# @:cmd
cat results.yml
