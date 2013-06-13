#
# Copyright (C) 2013, Will Dabney
#
# Licensed under the Apache License, Version 2.0 (the "License");
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
#
#  $Revision: 1 $
#  $Date: 2013-01-24 $
#  $Author: Will Dabney (amarack) $

from rlglue.agent.Agent import Agent
from rlglue.environment.Environment import Environment

from rlglue.types import Action
from rlglue.types import Observation

from rlglue.types import Observation_action
from rlglue.types import Reward_observation_action_terminal
from rlglue.types import Reward_observation_terminal

# This class provides a seemless way of running python RLGlue experiments locally without
# the use of sockets/network. I have no idea why this was not included in the python codec,
# but I really need this functionality. Maybe it will help you as well.
class LocalGlue:
	def __init__(self,theEnvironment,theAgent):
		self.env = theEnvironment
		self.agent = theAgent
		self.prevact = None
		self.reward_return = 0.0
		self.step_count = 0
		self.episode_count = 0
		self.exitStatus = 0

	def RL_init(self):
		taskSpecResponse = self.env.env_init()
		self.agent.agent_init(taskSpecResponse)
		self.prevact = None
		self.reward_return = 0.0
		self.step_count = 0
		self.episode_count = 0
		return taskSpecResponse

	def RL_start(self):
		self.reward_return = 0.0
		self.step_count = 1
		self.episode_count += 1
		self.prevact = None
		self.exitStatus = 0
		obs = self.env.env_start()
		action = self.agent.agent_start(obs)
		obsact = Observation_action()
		obsact.o = obs
		obsact.a = action
		self.prevact = action
		return obsact

	def RL_step(self):
		if self.prevact is None:
			self.RL_start()
		self.step_count += 1
		rot = self.env.env_step(self.prevact)
		roat = Reward_observation_action_terminal()
		roat.terminal = rot.terminal
		self.exitStatus = rot.terminal

		if rot.terminal == 1:
			self.agent.agent_end(rot.r)
			roat.a = self.prevact
			self.prevact = None
		else:
			self.prevact = self.agent.agent_step(rot.r, rot.o)
			roat.a = self.prevact

		self.reward_return += rot.r
		roat.r = rot.r
		roat.o = rot.o
		return roat

	def RL_cleanup(self):
		self.env.env_cleanup()
		self.agent.agent_cleanup()

	def RL_agent_message(self, message):
		if message == None:
			message=""
		return self.agent.agent_message(message)

	def RL_env_message(self, message):
		if message == None:
			message=""
		return self.env.env_message(message)

	def RL_return(self):
		return self.reward_return

	def RL_num_steps(self):
		return self.step_count

	def RL_num_episodes(self):
		return self.episode_count

	def RL_episode(self, num_steps):
		self.RL_start()
		while self.exitStatus != 1:
			# If num_steps is zero (or less) then treat as unlimited
			if (num_steps > 0) and self.step_count >= num_steps:
				break
			roat = self.RL_step()
			self.exitStatus = roat.terminal
		return self.exitStatus

