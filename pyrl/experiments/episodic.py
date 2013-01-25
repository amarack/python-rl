# 
# Copyright (C) 2008, Brian Tanner
# 
#http://rl-glue-ext.googlecode.com/
#
# Licensed under the Apache License, Version 2.0 (the "License"
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#



import sys
import timeit
import csv

import rlglue.RLGlue as rl_glue

from pyrl.rlglue import RLGlueLocal as RLGlueLocal
from pyrl.agents.skeleton_agent import skeleton_agent
from pyrl.environments.skeleton_environment import skeleton_environment


stepLimit = 5000
RLGlue = None

def setupExperiment(maxSteps, environment_class=None, agent_class=None):
	global stepLimit
	global RLGlue
	stepLimit = maxSteps
	if not (environment_class is None or agent_class is None):
		RLGlue = RLGlueLocal.LocalGlue(environment_class, agent_class)
	else:
		RLGlue = rl_glue

def runEpisode(timed=True):
	terminal = 0
	runtime = 0
	if timed:
		runtime = timeit.timeit('RLGlue.RL_episode(stepLimit)', setup = "from pyrl.experiments.episodic import RLGlue,stepLimit", number=1)
		terminal=RLGlue.exitStatus
	else:
		terminal=RLGlue.RL_episode(stepLimit)

	totalSteps=RLGlue.RL_num_steps()
	totalReward=RLGlue.RL_return()
	return terminal, totalSteps, totalReward, runtime


def runTrial(numEpisodes, filename=None, timed=True):
	taskSpec = RLGlue.RL_init()
	for i in range(numEpisodes):
		term,steps,reward, runtime = runEpisode(timed)
		if filename is None:
			print i, steps, runtime, reward, term
		else:
			with open(filename, "a") as f:
				csvwrite = csv.writer(f)
				csvwrite.writerow([i, steps, runtime, reward, term])
	RLGlue.RL_cleanup()



