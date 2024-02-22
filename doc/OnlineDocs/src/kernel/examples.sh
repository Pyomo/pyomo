#! /bin/bash

for file in `ls ../../library_reference/kernel/examples/*.py | sort`; do python $file; done;
