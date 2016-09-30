#!/bin/bash
# Usage: remove all utility bills pdf file password
shopt -s nullglob
for f in ../books-txt/*.txt
do
	python characterStats.py --file="$f" --benchmark --mincount=auto --mwclient="fr.wikipedia.org"
done
