#!/bin/bash

rm -f source/chemicalspace*.rst
sphinx-apidoc -o source -f ../chemicalspace
make html