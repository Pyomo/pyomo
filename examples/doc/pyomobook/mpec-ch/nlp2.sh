#!/bin/sh

# @pyomo:
pyomo solve --solver=mpec_nlp ex1a.py
# @:pyomo
cat results.yml
rm results.yml
