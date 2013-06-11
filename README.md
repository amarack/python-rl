python-rl
=========

Some Reinforcement Learning in Python


Run with:

python -m pyrl.rlglue.run

Many other run options exist. A good starting point is with the command line help:

python -m pyrl.rlglue.run --help

The params/ directory contains examples of experiments that demonstrate many of the different agent algorithms. 
As an example, a randomized trial experiment using mountain car, and a randomly generated 'fixed policy' can be 
run with:

python -m pyrl.rlglue.run --load params/mountaincar/example_randtrial.json

The out put of this particular experiment is of the form:
#evaluation points, list of evaluation index and evaluation value pairs, list of parameter values

For example:
1,0,-4999.0,0.0,0.219169344211,0.1,1.0,0.7,1,13709650200845

For this, there is oly 1 evaluation point (which is because this experiment only runs one episode). 
Then the evaluation index is zero, for the zero-th episode, followed by the return for that episode. 
Then we see a learning rate of 0.0 (because this is a fixed policy), followed by other parameters of 
Sarsa which in this case are not important. The final value of the line is the random seed used to 
generate the fixed policy.


Contributors
============
Will Dabney

Pierre-Luc Bacon
