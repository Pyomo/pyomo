#!/bin/sh

# @pyomo:
pyomo solve --solver=mpec_minlp \
            --solver-options="solver=glpk" ralph1.py
# @:pyomo
