#!/bin/sh

# @pyomo:
pyomo solve --solver=ipopt ex2.py
# @:pyomo
cat results.yml
