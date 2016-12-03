#!/bin/sh

# @pyomo:
pyomo solve --solver=ipopt \
            --transform=mpec.simple_nonlinear ex1a.py
# @:pyomo
