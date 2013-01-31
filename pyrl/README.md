pyRL
=========

I could rant all day long about the fact that most of the time Reinforcement Learning code 
available online tends to be completely broken, out of date, or very minimally useful. By 
far the biggest exception to this, in my opinion, has been the RL-Glue project. However, 
the project has either matured or been left on the shelf with few real updates in the last 
couple of years. 

pyRL is a project meant to provide an up to date collection of Reinforcement Learning 
agents, environments, and supporting methods written in Python, built on and extending the 
RL-Glue framework. Whenever possible it will make use of optimized python libraries such as 
numpy, scipy, scikits-learn, and neurolab. Some modules requiring additional speed will be 
written in C and will be compilable to Python modules. 

All agents and environments will be able to act as standalone RL-Glue network interfaces run 
from the commandline. However, pyRL also includes a module allowing agent, environment and experiment 
to be run together without the use of sockets. RL-Glue version 3.0 does not currently support that 
functionality for Python. 

This project is very much under development, but in the near-term I hope to have the most 
common RL environments, model-free and model-based agents implemented and working. From there 
I hope to add interesting new algorithms that I come across in the field (whenever I'm able to 
implement them successfully).

--Will Dabney
