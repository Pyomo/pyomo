#! /bin/bash

dir=`dirname $0`
for file in `ls ${dir}/examples/*.py | sort`; do python $file; done;
