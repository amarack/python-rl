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
class TWIP(Environment):
    """Two Wheel Inverted Pendulum (TWIP) robot simulation based upon the
    equations of motion derived in the paper:

    The Control of a Highly Nonlinear Two-wheels Balancing Robot:
        A Comparative Assessment between LQR and PID-PID Conrol Schemes. 2010.
    A.N.K. Nasir, M. A. Ahmad, and R.M.T. Raja Ismail.

    Currently also using their constants. This formulation is
    interesting because it includes yaw, and thus we can track 2D planar
    movement of the robot as well as its tilt. This might eventually
    lead to this environment having options for harder tasks in which
    the robot must balance AND navigate to some goal, etc. Currently
    it is only a balancing task.

    There are 9 actions which specify a voltage setting to each wheel motor of
    [-20., 0., 20.] volts. The model allows for continuous values but for now we
    have things discretized.

    This environment is still under development since there are constants in the equations
    whose values are only wild guesses, and the actual behavior of the system still needs to be
    checked/debugged.
    """

    name = "Two Wheel Inverted Pendulum"

    def __init__(self, **kwargs):
        self.noise = kwargs.setdefault('noise', 0.0)
        self.discount_factor = 1.0
        self.reward_range = (-1., 1.)

        self.x = numpy.zeros((2,))
        self.theta = numpy.zeros((2,))
        self.delta = numpy.zeros((2,))

        self.state_range = numpy.array([[-2., 2.],      # Linear Position
                                        [-10., -10.],   # Linear Pos. Velocity?
                                        [-numpy.pi * 50./180., numpy.pi * 50./180.],    # Angle from vertical
                                        [-numpy.pi * 2.5, numpy.pi * 2.5],              # Ang. Vel. vertical
                                        [-numpy.pi * 2.5, numpy.pi * 2.5],              # pos. angle
                                        [-numpy.pi * 2.5, numpy.pi * 2.5]])             # pos. ang. velocity

        self.delta_time = 0.01
        self.sim_steps = 10
        self.max_volt = 20.

        self.gravity = 9.81
        self.D = 0.2            # Dist (meters) between contact patches of wheels
        self.J_p = 0.0041       # Chassis inertia (kg . m^2)
        self.J_pd = 0.00018     # Chassis inertia during rotation (kg . m^2)
        self.J_w = 0.000039     # Wheel's inertia (kg . m^2)
        self.k_e = 0.006087     # Back EMF constant (Vs/rad)
        self.k_m = 0.006123     # Motor torque constant (Nm/A)
        self.l = 0.07           # Dist between center of wheels and robots center of grav. (m)
        self.M_p = 1.13         # Body's mass (kg)
        self.M_w = 0.03         # Wheel's mass (kg)
        self.R = 3.             # Nominal Terminal resistance (omhs)
        self.r = 0.051          # Wheel's radius (m)

        self.f_drR = 0.01      # Unknown? Not mentioned in the paper anywhere as to what these are
        self.f_drL = 0.01
        self.f_dp = 0.01

        # Useful quantities
        self.alpha = 2. * self.M_w + (2.*self.J_w/self.r**2) + self.M_p
        self.gamma = self.J_p + self.M_p * self.l**2


    def makeTaskSpec(self):
        ts = TaskSpecRLGlue.TaskSpec(discount_factor=self.discount_factor,
                                    reward_range=self.reward_range)
        ts.setDiscountFactor(self.discount_factor)
        ts.addDiscreteAction((0, 8))
        for minValue, maxValue in self.state_range:
            ts.addContinuousObservation((minValue, maxValue))
        ts.setEpisodic()
        ts.setExtra(self.name)
        return ts.toTaskSpec()

    def getState(self):
        return self.x.tolist() + self.theta.tolist() + self.delta.tolist()

    def reset(self):
        self.x.fill(0.0)
        self.theta.fill(0.0)
        self.delta.fill(0.0)
        self.theta[0] = numpy.pi * 1./180.

    def env_init(self):
        return self.makeTaskSpec()

    def env_start(self):
        self.reset()
        returnObs = Observation()
        returnObs.doubleArray = self.getState()
        return returnObs

    def takeAction(self, intAction):
        V_left = self.max_volt * ((intAction / 3) - 1)
        V_right = self.max_volt * ((intAction % 3) - 1)
        if self.noise > 0:
            V_left += self.max_volt * numpy.random.normal(scale=self.noise)
            V_right += self.max_volt * numpy.random.normal(scale=self.noise)

        for step in range(self.sim_steps):
            sinO = numpy.sin(self.theta[0])
            cosO = numpy.cos(self.theta[0])
            beta = (self.alpha*self.gamma - self.M_p**2 * self.l**2 * cosO**2) / (self.alpha * self.gamma)

            # Terms that repear multiple times
            term1 = (self.k_m/(self.alpha * self.R * beta))
            term2 = ((1./self.r) + (self.M_p * self.l * cosO)/self.gamma)

            x_accel = -2. * self.k_e * term1 * term2 * self.x[1]
            x_accel += self.theta[0] * (self.M_p**2 * self.gravity * self.l**2 * sinO * cosO)/(self.alpha * self.gamma * beta * self.theta[0])
            x_accel += term1 * term2 * (V_right + V_left)
            x_accel += (self.f_drR + self.f_drL)/(self.alpha * beta)
            x_accel += self.f_dp * (1. + (self.M_p * self.l**2 * cosO**2)/self.gamma)/(self.alpha * beta)
            x_accel += (self.M_p * self.l * self.theta[1]**2 * sinO)/(self.alpha * beta)

            term3 = (1. + (self.M_p * self.l * cosO)/(self.alpha * self.r))
            theta_accel = ((2. * self.k_e * self.k_m)/(self.gamma * self.r * self.R * beta)) * term3 * self.x[1]
            theta_accel += ((self.M_p * self.gravity * self.l * sinO)/(self.gamma * beta * self.theta[0]))*self.theta[0]
            theta_accel -= (self.k_m/(self.gamma * self.R * beta)) * term3 * (V_right + V_left)
            theta_accel += ((self.M_p * self.l * cosO)/(self.alpha * self.gamma * beta)) * (-self.f_drR + self.f_drL)
            theta_accel += (self.l * cosO / (self.gamma * beta)) * (1. - self.M_p/self.alpha) * self.f_dp
            theta_accel -= (self.M_p**2 * self.l**2 * self.theta[1]**2 * sinO * cosO)/(self.alpha * self.gamma * beta)

            term4 = (self.D)/(2. * self.J_pd)
            delta_accel = (self.k_m/(self.r * self.R))*term4 * (V_left - V_right)
            delta_accel += term4 * (self.f_drL - self.f_drR)

            # Update state variables
            df = (self.delta_time / float(self.sim_steps))
            self.x += df * numpy.array([self.x[1], x_accel])
            self.theta += df * numpy.array([self.theta[1], theta_accel])
            self.delta += df * numpy.array([self.delta[1], delta_accel])

        if self.terminate():
            return -1., True
        else:
            return 1., False

    def terminate(self):
        """Indicates whether or not the episode should terminate.

        Returns:
            A boolean, true indicating the end of an episode and false indicating the episode should continue.
            False is returned if either the cart location or
            the pole angle is beyond the allowed range.
        """
        return numpy.abs(self.x[0]) > self.state_range[0,1] or (numpy.abs(self.theta[0]) > self.state_range[2,1]).any()

    def env_step(self,thisAction):
        intAction = thisAction.intArray[0]

        theReward, episodeOver = self.takeAction(intAction)

        theObs = Observation()
        theObs.doubleArray = self.getState()
        returnRO = Reward_observation_terminal()
        returnRO.r = theReward
        returnRO.o = theObs
        returnRO.terminal = int(episodeOver)
        return returnRO

    def env_cleanup(self):
        pass

    def env_message(self,inMessage):
        return "I don't know how to respond to your message";
