#!/bin/sh

pyomo solve --solver=ipopt --summary --stream-solver ReactorDesign.py
