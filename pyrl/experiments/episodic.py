
# Author: Will Dabney
# Author: Pierre-Luc Bacon <pierrelucbacon@gmail.com>

import csv, os
from pyrl.misc.timer import Timer
from pyrl.rlglue import RLGlueLocal as RLGlueLocal
from pyrl.rlglue.registry import register_experiment
import rlglue.RLGlue as rl_glue
import pyrl.visualizers.plotExperiment as plotExperiment

@register_experiment
class Episodic(object):
    name = "Episodic"

    def __init__(self, config, **kwargs):
        self.maxsteps = kwargs.setdefault('maxsteps', 5000)
        self.num_episodes = kwargs.setdefault('num_episodes', 10)
        self.num_runs = kwargs.setdefault('num_runs', 1)
        self.timed = kwargs.setdefault('timed', True)
        self.configuration = config

        if kwargs.has_key('agent') and kwargs.has_key('environment'):
            self.agent = kwargs['agent']
            self.environment = kwargs['environment']
            self.rlglue = RLGlueLocal.LocalGlue(self.environment, self.agent)
        else:
            self.rlglue = rl_glue

    def run_episode(self):
        terminal = 0
        runtime = 0
        # Query the agent whether or not it has diverged
        if self.hasAgentDiverged():
            return 0, -1, 0.0, 0.0 # -1 number of steps, signals that divergence.
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

    def run_experiment(self, filename=None):
        if filename is None:
            print 'trial, number of steps, runtime, accumulated reward, termination'
        for run in range(self.num_runs):
            self.run_trial(filename=filename)

    def hasAgentDiverged(self):
        """Sends an rl-glue message to the agent asking if it has diverged or not.
        The message is exactly: agent_diverged?
        The expected response is: True (if it has), False (if it has not)
        The responses are not case sensitive, and anything other than true or false
        will be treated as a false (to support agents which do not have this implemented).
        """
        return self.rlglue.RL_agent_message("agent_diverged?").lower() == "true"







