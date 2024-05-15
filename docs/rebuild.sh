rm -f chemicalspace*.rst modules.rst
sphinx-apidoc --ext-autodoc --ext-coverage --ext-mathjax -o . -f ../chemicalspace
make html