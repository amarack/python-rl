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
    """Cart Pole environment. This implementation alows multiple poles,
    noisy action, and random starts. It has been checked repeatedly for
    'correctness', specifically the direction of gravity. Some implementations of
    cart pole on the internet have the gravity constant inverted. The way to check is to
    limit the force to be zero, start from a valid random start state and watch how long
    it takes for the pole to fall. If the pole falls almost immediately, you're all set. If it takes
    tens or hundreds of steps then you have gravity inverted. It will tend to still fall because
    of round off errors that cause the oscillations to grow until it eventually falls.
    """
    name = "Cart Pole"

    def __init__(self, mode='easy', pole_scales=[1.], noise=0.0, reward_noise=0.0, random_start=True):
        self.noise = noise
        self.reward_noise = reward_noise
        self.random_start = random_start
        self.cart_location = 0.0
        self.cart_velocity = 0.0
        self.pole_angle = numpy.zeros((len(pole_scales),))
        self.pole_velocity = numpy.zeros((len(pole_scales),))

        # Setup pole lengths and masses based on scale of each pole
        # (Papers using multi-poles tend to have them either same lengths/masses
        #   or they vary by some scalar from the other poles)
        pole_scales = numpy.array(pole_scales)
        self.pole_length = numpy.ones((len(pole_scales),))*0.5 * pole_scales
        self.pole_mass = numpy.ones((len(pole_scales),))*0.1 * pole_scales

        self.domain_name = "Cart Pole"

        self.mode = mode
        if mode == 'hard':
            self.state_range = numpy.array([[-3., 3.],                                   # Cart location bound
                                            [-5., 5.],                                    # Cart velocity bound
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

        self.reward_range = (-1000., 1.*len(pole_scales)) if self.mode == "swingup" else (-1., 1.)
        self.delta_time = 0.02
        self.max_force = 10.
        self.gravity = -9.8
        self.cart_mass = 1.

    def makeTaskSpec(self):
        ts = TaskSpecRLGlue.TaskSpec(discount_factor=self.discount_factor,
                                    reward_range=self.reward_range)
        ts.setDiscountFactor(self.discount_factor)
        ts.addDiscreteAction((0, 1))
        for minValue, maxValue in self.state_range:
            ts.addContinuousObservation((minValue, maxValue))
        for extra_poles in range(len(self.pole_angle)-1):
            for minValue, maxValue in self.state_range[-2:]:
                ts.addContinuousObservation((minValue, maxValue))
        ts.setEpisodic()
        ts.setExtra(self.domain_name)
        return ts.toTaskSpec()

    def reset(self):
        self.cart_location = 0.0
        self.cart_velocity = 0.0
        self.pole_angle.fill(0.0)
        self.pole_velocity.fill(0.0)
        if self.random_start:
            self.pole_angle = (numpy.random.random(self.pole_angle.shape)-0.5)/5.

    def env_init(self):
        return self.makeTaskSpec()

    def env_start(self):
        self.reset()
        returnObs = Observation()
        returnObs.doubleArray = [self.cart_location, self.cart_velocity] + self.pole_angle.tolist() + self.pole_velocity.tolist()
        return returnObs

    def __gravity_on_pole(self):
        pull = self.mu_p * self.pole_velocity/(self.pole_mass * self.pole_length)
        pull += self.gravity * numpy.sin(self.pole_angle)
        return pull

    def __effective_force(self):
        F = self.pole_mass * self.pole_length * self.pole_velocity**2 * numpy.sin(self.pole_angle)
        F += .75 * self.pole_mass * numpy.cos(self.pole_angle) * self.__gravity_on_pole()
        return F.sum()

    def __effective_mass(self):
        return (self.pole_mass * (1. - .75 * numpy.cos(self.pole_angle)**2)).sum()

    def takeAction(self, intAction):
        force = self.max_force if intAction == 1 else -self.max_force
        force += self.max_force*numpy.random.normal(scale=self.noise) if self.noise > 0 else 0.0 # Compute noise

        for step in range(self.sim_steps):
            cart_accel = force - self.mu_c * numpy.sign(self.cart_velocity) + self.__effective_force()
            cart_accel /= self.cart_mass + self.__effective_mass()
            pole_accel = (-.75/self.pole_length) * (cart_accel * numpy.cos(self.pole_angle) + self.__gravity_on_pole())

            # Update state variables
            df = (self.delta_time / float(self.sim_steps))
            self.cart_location += df * self.cart_velocity
            self.cart_velocity += df * cart_accel
            self.pole_angle += df * self.pole_velocity
            self.pole_velocity += df * pole_accel

        # If theta (state[2]) has gone past our conceptual limits of [-pi,pi]
        # map it onto the equivalent angle that is in the accepted range (by adding or subtracting 2pi)
        for i in range(len(self.pole_angle)):
            while self.pole_angle[i] < -numpy.pi:
                self.pole_angle[i] += 2. * numpy.pi

            while self.pole_angle[i] > numpy.pi:
                self.pole_angle[i] -= 2. * numpy.pi

        if self.mode == 'swingup':
            return numpy.cos(numpy.abs(self.pole_angle)).sum()
        else:
            return -1. if self.terminate() else 1.

    def terminate(self):
        """Indicates whether or not the episode should terminate.

        Returns:
            A boolean, true indicating the end of an episode and false indicating the episode should continue.
            False is returned if either the cart location or
            the pole angle is beyond the allowed range.
        """
        return numpy.abs(self.cart_location) > self.state_range[0,1] or (numpy.abs(self.pole_angle) > self.state_range[2,1]).any()

    def env_step(self,thisAction):
        intAction = thisAction.intArray[0]

        theReward = self.takeAction(intAction)
        episodeOver = int(self.terminate())

        if self.reward_noise > 0:
            theReward += numpy.random.normal(scale=self.reward_noise)

        theObs = Observation()
        theObs.doubleArray = [self.cart_location, self.cart_velocity] + self.pole_angle.tolist() + self.pole_velocity.tolist()
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

