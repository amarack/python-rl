#
# Copyright (C) 2007, Mark Lee
#
#http://rl-glue-ext.googlecode.com/
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

import random
from rlglue.environment.Environment import Environment
from rlglue.environment import EnvironmentLoader as EnvironmentLoader
from rlglue.types import Observation
from rlglue.types import Action
from rlglue.types import Reward_observation_terminal
from pyrl.rlglue.registry import register_environment

# /**
#  *  This is a very simple environment with discrete observations corresponding to states labeled {0,1,...,19,20}
#     The starting state is 10.
#
#     There are 2 actions = {0,1}.  0 decrements the state, 1 increments the state.
#
#     The problem is episodic, ending when state 0 or 20 is reached, giving reward -1 or +1, respectively.  The reward is 0 on
#     all other steps.
#  * @author Brian Tanner
#  */

@register_environment
class skeleton_environment(Environment):
    name = "Skeleton environment"

    currentState=10
    def env_init(self):
        return "VERSION RL-Glue-3.0 PROBLEMTYPE episodic DISCOUNTFACTOR 1.0 OBSERVATIONS INTS (0 20)  ACTIONS INTS (0 1)  REWARDS (-1.0 1.0)  EXTRA skeleton_environment(Python) by Brian Tanner."

    def env_start(self):
        self.currentState=10

        returnObs=Observation()
        returnObs.intArray=[self.currentState]

        return returnObs

    def env_step(self,thisAction):
        episodeOver=0
        theReward=0

        if    thisAction.intArray[0]==0:
            self.currentState=self.currentState-1
        if    thisAction.intArray[0]==1:
            self.currentState=self.currentState+1

        if self.currentState <= 0:
            self.currentState=0
            theReward=-1
            episodeOver=1

        if self.currentState >= 20:
            self.currentState=20
            theReward=1
            episodeOver=1

        theObs=Observation()
        theObs.intArray=[self.currentState]

        returnRO=Reward_observation_terminal()
        returnRO.r=theReward
        returnRO.o=theObs
        returnRO.terminal=episodeOver

        return returnRO

    def env_cleanup(self):
        pass

    def env_message(self,inMessage):
        if inMessage=="what is your name?":
            return "my name is skeleton_environment, Python edition!";
        else:
            return "I don't know how to respond to your message";


if __name__=="__main__":
    EnvironmentLoader.loadEnvironment(skeleton_environment())
