#!/bin/bash
# Usage: remove all utility bills pdf file password
shopt -s nullglob
for f in ../books-txt/*.txt
do
	python characterStats-averages.py --file="$f" --benchmark --avg=$1
	#--avg="equal,confident"
	#--avg="equal"
	#--avg="meta"
done