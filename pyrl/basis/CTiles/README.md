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
cmake .
make
... this creates the tiles.so and tiles.o files


Note About CMake and Python on Mac: 
For some reason things can sometimes get messed up with this combination. Some people claim 
this is a bug in cmake or a bug from mac. It comes up when you have multiple python distributions 
installed. So, most people should be fine, but if you get a fatal error when trying to use this 
module in python you should look into uninstalling the unused distributions or pass to cmake 
the following arguments with the correct values filled in: 

-DPYTHON_LIBRARY=... -DPYTHON_INCLUDE=...

