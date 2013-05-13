
# Author: Will Dabney
# Author: Pierre-Luc Bacon <pierrelucbacon@gmail.com>

import csv
from pyrl.misc.timer import Timer
from pyrl.rlglue import RLGlueLocal as RLGlueLocal
from pyrl.rlglue.registry import register_experiment
import rlglue.RLGlue as rl_glue
import pyrl.visualizers.plotExperiment as plotExperiment

@register_experiment
class Episodic(object):
    name = "Episodic"

    def __init__(self, agent=None, environment=None, maxsteps=5000, num_episodes=10,
                 num_runs=1, timed=True, evaluate='reward'):
        self.maxsteps = maxsteps
        self.num_episodes = num_episodes
        self.num_runs = num_runs
        self.timed = timed
        self.evaluate = evaluate

        if agent is not None and environment is not None:
            self.rlglue = RLGlueLocal.LocalGlue(environment, agent)
            self.agent = agent
            self.environment = environment
        else:
            self.rlglue = rl_glue

    def run_episode(self):
        terminal = 0
        runtime = 0
        if self.timed:
            timer = Timer()
            with timer:
                terminal = self.rlglue.RL_episode(self.maxsteps)
            runtime = timer.duration_in_seconds()

        else:
            terminal = self.rlglue.RL_episode(self.maxsteps)

        totalSteps = self.rlglue.RL_num_steps()
        totalReward = self.rlglue.RL_return()
        return terminal, totalSteps, totalReward, runtime

    def run_trial(self, filename=None):
        self.rlglue.RL_init()
        for i in range(self.num_episodes):
            term, steps, reward, runtime = self.run_episode()
            if filename is None:
                print i, steps, runtime, reward, term
            else:
                with open(filename, "a") as f:
                    csvwrite = csv.writer(f)
                    csvwrite.writerow([i, steps, runtime, reward, term])
        self.rlglue.RL_cleanup()

    def run_experiment(self, filename=None, **args):
        if filename is None:
            print 'trial, number of steps, runtime, accumulated reward, termination'
        for run in range(self.num_runs):
            self.run_trial(filename=filename)




