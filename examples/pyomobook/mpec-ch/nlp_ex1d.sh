#!/bin/sh

# @pyomo:
pyomo solve --solver=mpec_nlp ex1d.py
# @:pyomo
cat results.yml
rm results.yml
