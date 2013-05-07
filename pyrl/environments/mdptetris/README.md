mdptetris
=========

This is a modified version of mdptetris from:

   	https://gforge.inria.fr/projects/mdptetris/
	http://mdptetris.gforge.inria.fr/doc/

It extends their Tetris implementation to be compiled as a python module, and adds 
a few utility functions for use with general reinforcement learning agents. We have 
also removed most of the agent logic from the original code, and are continuing to 
pair down this fork of their codebase to just the essentials for use as a Tetris 
reinforcement learning environment, where agents do not have specific domain knowledge.
