#!/bin/sh

# @pyomo:
pyomo solve --solver=mpec_nlp ex1b.py
# @:pyomo
cat results.yml
rm results.yml
