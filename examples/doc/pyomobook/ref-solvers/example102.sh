#!/bin/sh

# @cmd:
pyomo solve --solver=ipopt simple.py
# @:cmd
cat results.yml
