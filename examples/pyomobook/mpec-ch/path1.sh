#!/bin/sh

# @pyomo:
pyomo solve --solver=path munson1.py
# @:pyomo
rm -f results.yml
