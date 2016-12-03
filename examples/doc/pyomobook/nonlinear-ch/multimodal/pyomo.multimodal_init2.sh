#!/bin/sh

pyomo solve --solver=ipopt --summary multimodal_init2.py
rm -f results.yml
