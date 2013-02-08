pyrl.environments.libPOMDP
============================

This is primarily code from pomdp-solve written by Anthony R. Cassandra. I've only added a 
new makefile, which is not as sophisticated as the original, and some code to allow the whole thing 
to be compiled into a Python module.

At present this only contains the code relevant for reading and writing the MDP/POMDP specification 
files, and interacting with the information contained within them. However, pomdp-solve itself 
has many useful implementations in pure C that may later be brought into this module for use in python.

This has been tested on Mac OS X, but 'should' also work in Linux. 