#!/bin/sh

# @cmd:
pyomo solve --solver=asl:ipopt simple.py
# @:cmd
cat results.yml
