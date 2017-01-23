#!/bin/sh

# @pyomo:
pyomo solve --model-name=transformed --solver=ipopt ex2.py
# @:pyomo
cat results.yml
rm results.yml
