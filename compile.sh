#!/bin/bash

python2 toy_example.py
python2 TEM_example.py
pdflatex DMD_Clustering.tex
bibtex DMD_Clustering.aux
pdflatex DMD_Clustering.tex
pdflatex DMD_Clustering.tex


