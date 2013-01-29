pyrl.basis.CTiles
====================

This folder contains the C version of tile coding, as well as the python routines which call the c version of tiles.
This was written by Rich Sutton and only the makefile has changed from the original.

The following files are here:

Makefile - compiles both the C version and the Python->C version (for Mac or Linux)
tiles.h - header for C version of tiles
tiles.cpp - c++ version of tiles
tiletimes.cpp - timing code for c calling c version of tiles
tilesInt.C - interface so that Python can call the c version
tiletimes.py - timing code for the python calling c version of tiles
fancytiles.py - code to get different shapes and sizes of tiles

To use these:
In a terminal window:
make
... this creates the tiles.so and tiles.o files

