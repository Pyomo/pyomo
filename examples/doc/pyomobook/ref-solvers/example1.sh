#!/bin/sh

# @cmd:
pyomo solve --solver=cbc simple.py
# @:cmd
cat results.yml
