#!/bin/sh

# @pyomo:
pyomo solve --solver=mpec_nlp \
            --solver-options="epsilon_initial=0.1 \
                epsilon_final=1e-7" \
            ex1a.py
# @:pyomo
