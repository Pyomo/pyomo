#!/bin/sh

# @pyomo:
pyomo solve --solver=mpec_nlp ex1e.py
# @:pyomo
cat results.yml
rm results.yml
