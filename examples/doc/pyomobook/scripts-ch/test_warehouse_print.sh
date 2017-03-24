#!/bin/sh

# Note: the system sort is not consistent among platforms or users (in
# particular, how numbers and strings are sorted).  So we will use
# something consistent: Python!

python warehouse_print.py \
    | python -c "import sys; sys.stdout.write(''.join(sorted(sys.stdin)))"
