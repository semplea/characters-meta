#!/bin/bash
# Usage: remove all utility bills pdf file password
shopt -s nullglob
python characterStats.py --file=../books-txt/madamebovary.txt --benchmark --mincount=10
python characterStats.py --file=../books-txt/notredamedeparis.txt --benchmark --mincount=10
python characterStats.py --file=../books-txt/letourdumondeen80jours.txt --benchmark --mincount=10