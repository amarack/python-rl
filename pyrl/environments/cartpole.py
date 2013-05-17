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

import numpy

from rlglue.environment.Environment import Environment
from rlglue.environment import EnvironmentLoader as EnvironmentLoader
from rlglue.types import Observation
from rlglue.types import Action
from rlglue.types import Reward_observation_terminal
from pyrl.rlglue import TaskSpecRLGlue
from pyrl.rlglue.registry import register_environment

@register_environment
class CartPole(Environment):
	name = "Cart Pole"

	def __init__(self, mode='easy', noise=0.0, random_start=False):
            self.noise = noise
            self.random_start = random_start
            self.state = numpy.zeros((4,))
            self.domain_name = "Cart Pole"

	    self.mode = mode
	    if mode == 'hard':
		    self.state_range = numpy.array([[-3., 3.],                                   # Cart location bound
						    [-5., 5.],                                   # Cart velocity bound
						    [-numpy.pi * 45./180., numpy.pi * 45./180.], # Pole angle bounds
						    [-2.5*numpy.pi, 2.5*numpy.pi]])              # Pole velocity bound
		    self.mu_c = 0.0005
		    self.mu_p = 0.000002
		    self.sim_steps = 10
		    self.discount_factor = 0.999
	    elif mode == 'swingup':
		    self.state_range = numpy.array([[-3., 3.],                                   # Cart location bound
						    [-5., 5.],                                   # Cart velocity bound
						    [-numpy.pi, numpy.pi],                       # Pole angle bounds
						    [-2.5*numpy.pi, 2.5*numpy.pi]])              # Pole velocity bound
		    self.mu_c = 0.0005
		    self.mu_p = 0.000002
		    self.sim_steps = 10
		    self.discount_factor = 1.
	    else:
		    if mode != 'easy':
			    print "Error: CartPole does not recognize mode", mode
			    print "Defaulting to 'easy'"
		    self.state_range = numpy.array([[-2.4, 2.4],                                 # Cart location bound
						    [-6., 6.],                                   # Cart velocity bound
						    [-numpy.pi * 12./180., numpy.pi * 12./180.], # Pole angle bounds
						    [-6., 6.]])                                  # Pole velocity bound
		    self.mu_c = 0.
		    self.mu_p = 0.
		    self.sim_steps = 1	    
		    self.discount_factor = 0.999

	    self.reward_range = (-1000., 1.) if self.mode == "swingup" else (-1., 1.)
            self.delta_time = 0.02
	    self.max_force = 10.
            self.gravity = 9.8
	    self.pole_length = 0.5
	    self.pole_mass = 0.1
	    self.cart_mass = 1.
	    

	def makeTaskSpec(self):
		ts = TaskSpecRLGlue.TaskSpec(discount_factor=self.discount_factor, 
					     reward_range=self.reward_range)
                ts.setDiscountFactor(self.discount_factor)
		ts.addDiscreteAction((0, 1))
                for minValue, maxValue in self.state_range:
                    ts.addContinuousObservation((minValue, maxValue))

		ts.setEpisodic()
		ts.setExtra(self.domain_name)
		return ts.toTaskSpec()

	def reset(self):
		self.state.fill(0.)
		if self.random_start:
                    self.state[2] = (numpy.random.random()-0.5)/5.

	def env_init(self):
		return self.makeTaskSpec()
	
	def env_start(self):
            self.reset()
            returnObs = Observation()
            returnObs.doubleArray = self.state.tolist()
            return returnObs

        def takeAction(self, intAction):
            force = self.max_force if intAction == 1 else -self.max_force
            force += self.max_force*numpy.random.normal(scale=self.noise) if self.noise > 0 else 0.0 # Compute noise

	    for step in range(self.sim_steps):
                # Precompute sin(theta) and cos(theta)
                costheta = numpy.cos(self.state[2])
		sintheta = numpy.sin(self.state[2])
                # This term appears in equations for acceleration
		temp = (force + self.pole_length*self.pole_mass * self.state[3]**2 * sintheta - self.mu_c*numpy.sign(self.state[1])) / (self.cart_mass + self.pole_mass)
		temp += (self.mu_p * self.state[3])/(self.pole_length * self.pole_mass * costheta)
		# Compute acceleration for pole angle and cart
		theta_acc = (self.gravity * sintheta - costheta * temp) / (self.pole_length * ((4./3.) - self.pole_mass * costheta**2 / (self.cart_mass + self.pole_mass)))
		cart_acc = temp - self.pole_length * self.pole_mass * theta_acc * costheta / (self.cart_mass + self.pole_mass)

		# Update state variables
		self.state += (self.delta_time / float(self.sim_steps)) * numpy.array([self.state[1], cart_acc, self.state[3], theta_acc])

		# If theta (state[2]) has gone past our conceptual limits of [-pi,pi]
		# map it onto the equivalent angle that is in the accepted range (by adding or subtracting 2pi)
		while self.state[2] < -numpy.pi:
                    self.state[2] += 2. * numpy.pi

		while self.state[2] > numpy.pi:
                    self.state[2] -= 2. * numpy.pi

	    if self.mode == 'swingup':
                return numpy.cos(numpy.abs(self.state[2]))
	    else:
                return -1. if self.terminate() else 1.

	def terminate(self):
            """Indicates whether or not the episode should terminate.

	    Returns:
	        A boolean, true indicating the end of an episode and false indicating the episode should continue.
		False is returned if either the cart location or 
		the pole angle is beyond the allowed range.
            """

            return (numpy.abs(self.state[(0,2),:]) > self.state_range[(0,2),1]).any()
	
	def env_step(self,thisAction):
		intAction = thisAction.intArray[0]
		
		theReward = self.takeAction(intAction)
		episodeOver = int(self.terminate())

		theObs = Observation()
		theObs.doubleArray = self.state.tolist()
		
		returnRO = Reward_observation_terminal()
		returnRO.r = theReward
		returnRO.o = theObs
		returnRO.terminal = episodeOver

		return returnRO

	def env_cleanup(self):
		pass

	def env_message(self,inMessage):
		return "I don't know how to respond to your message";


if __name__=="__main__":
	import argparse
	parser = argparse.ArgumentParser(description='Run Noisy Cart Pole Balancing or Swing Up environment in network mode.')
	parser.add_argument("--noise", type=float, default=0, help="Standard deviation of additive noise to generate, affects the action effects.")
	parser.add_argument("--random_restarts", type=bool, default=False, help="Restart the cart with a random location and velocity.")
	parser.add_argument("--mode", choices=["easy", "hard", "swingup"], default="easy", type=str,
			    help="Choose the type of cart pole domain. Easy/hard balancing, or swing up.")


	args = parser.parse_args()
	EnvironmentLoader.loadEnvironment(CartPole(mode=args.mode, noise=args.noise, random_start=args.random_restarts))

