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
from pyrl.misc import matrix

@register_environment
class Bicycle(Environment):
    """Bicycle balancing/riding domain.

    From the paper:
    Learning to Drive a Bicycle using Reinforcement Learning and Shaping.
    Jette Randlov and Preben Alstrom. 1998.
    """

    name = "Bicycle"

    def __init__(self, **kwargs):
        self.noise = kwargs.setdefault('noise', 0.04)
        self.random_start = kwargs.setdefault('random_start', False)

        self.state = numpy.zeros((5,)) # omega, omega_dot, omega_ddot, theta, theta_dot
        self.position = numpy.zeros((5,)) # x_f, y_f, x_b, y_b, psi
        self.state_range = numpy.array([[-numpy.pi * 12./180., numpy.pi * 12./180.],
                                        [-numpy.pi * 2./180., numpy.pi * 2./180.],
                                        [-numpy.pi, numpy.pi],
                                        [-numpy.pi * 80./180., numpy.pi * 80./180.],
                                        [-numpy.pi * 2./180., numpy.pi * 2./180.]])
        self.psi_range = numpy.array([-numpy.pi, numpy.pi])

        self.reward_fall = -1.0
        self.reward_goal = 0.01
        self.goal_rsqrd = 100.0 # Square of the radius around the goal (10m)^2
        self.navigate = kwargs.setdefault('navigate', False)
        if not self.navigate:
            # Original balancing task
            self.reward_shaping = 0.001
        else:
            self.reward_shaping = 0.00004

        self.goal_loc = numpy.array([1000., 0])

        # Units in Meters and Kilograms
        self.c = 0.66       # Horizontal dist between bottom of front wheel and center of mass
        self.d_cm = 0.30    # Vertical dist between center of mass and the cyclist
        self.h = 0.94       # Height of the center of mass over the ground
        self.l = 1.11       # Dist between front tire and back tire at point on ground
        self.M_c = 15.0     # Mass of bicycle
        self.M_d = 1.7      # Mass of tire
        self.M_p = 60       # Mass of cyclist
        self.r = 0.34       # Radius of tire
        self.v = 10.0 / 3.6 # Velocity of bicycle (converted from km/h to m/s)

        # Useful precomputations
        self.M = self.M_p + self.M_c
        self.Inertia_bc = (13./3.) * self.M_c * self.h**2 + self.M_p * (self.h + self.d_cm)**2
        self.Inertia_dv = self.M_d * self.r**2
        self.Inertia_dl = .5 * self.M_d * self.r**2
        self.sigma_dot = self.v / self.r

        # Simulation Constants
        self.gravity = 9.8
        self.delta_time = 0.02
        self.sim_steps = 10


    def makeTaskSpec(self):
        ts = TaskSpecRLGlue.TaskSpec(discount_factor=1.0, reward_range=(-1.0, 1.0))
        ts.addDiscreteAction((0, 8)) # 9 actions
        for minValue, maxValue in self.state_range:
            ts.addContinuousObservation((minValue, maxValue))
        ts.setEpisodic()
        ts.setExtra(self.name)
        return ts.toTaskSpec()

    def reset(self):
        self.state.fill(0.0)
        self.position.fill(0.0)
        self.position[3] = self.l
        self.position[4] = numpy.arctan((self.position[1]-self.position[0])/(self.position[2] - self.position[3]))

    def env_init(self):
        return self.makeTaskSpec()

    def env_start(self):
        self.reset()
        returnObs = Observation()
        returnObs.doubleArray = self.state.tolist()
        return returnObs

    def takeAction(self, intAction):
        T = 2. * ((int(intAction)/3) - 1) # Torque on handle bars
        d = 0.02 * ((intAction % 3) - 1) # Displacement of center of mass (in meters)
        if self.noise > 0:
            d += (numpy.random.random()-0.5)*self.noise # Noise between [-0.02, 0.02] meters

        omega, omega_dot, omega_ddot, theta, theta_dot = tuple(self.state)
        x_f, y_f, x_b, y_b, psi = tuple(self.position)

        for step in range(self.sim_steps):
            if theta == 0: # Infinite radius tends to not be handled well
                r_f = r_b = r_CM = 1.e8
            else:
                r_f = self.l / numpy.abs(numpy.sin(theta))
                r_b = self.l / numpy.abs(numpy.tan(theta))
                r_CM = numpy.sqrt((self.l - self.c)**2 + (self.l**2 / numpy.tan(theta)**2))

            varphi = omega + numpy.arctan(d / self.h)

            omega_ddot = self.h * self.M * self.gravity * numpy.sin(varphi)
            omega_ddot -= numpy.cos(varphi) * (self.Inertia_dv * self.sigma_dot * theta_dot + numpy.sign(theta)*self.v**2*(self.M_d * self.r *(1./r_f + 1./r_b) + self.M*self.h/r_CM))
            omega_ddot /= self.Inertia_bc

            theta_ddot = (T - self.Inertia_dv * self.sigma_dot * omega_dot) / self.Inertia_dl

            df = (self.delta_time / float(self.sim_steps))
            omega_dot += df * omega_ddot
            omega += df * omega_dot
            theta_dot += df * theta_ddot
            theta += df * theta_dot

            # Handle bar limits (80 deg.)
            theta = numpy.clip(theta, self.state_range[3,0], self.state_range[3,1])

            # Update position (x,y) of tires
            front_term = psi + theta + numpy.sign(psi + theta)*numpy.arcsin(self.v * df / (2.*r_f))
            back_term = psi + numpy.sign(psi)*numpy.arcsin(self.v * df / (2.*r_b))
            x_f += -numpy.sin(front_term)
            y_f += numpy.cos(front_term)
            x_b += -numpy.sin(back_term)
            y_b += numpy.cos(back_term)

            # Handle Roundoff errors, to keep the length of the bicycle constant
            dist = numpy.sqrt((x_f-x_b)**2 + (y_f-y_b)**2)
            if numpy.abs(dist - self.l) > 0.01:
                x_b += (x_b - x_f) * (self.l - dist)/dist
                y_b += (y_b - y_f) * (self.l - dist)/dist

            # Update psi
            if x_f==x_b and y_f-y_b < 0:
                psi = numpy.pi
            elif y_f - y_b > 0:
                psi = numpy.arctan((x_b - x_f)/(y_f - y_b))
            else:
                psi = numpy.sign(x_b - x_f)*(numpy.pi/2.) - numpy.arctan((y_f - y_b)/(x_b-x_f))

        self.state = numpy.array([omega, omega_dot, omega_ddot, theta, theta_dot])
        self.position = numpy.array([x_f, y_f, x_b, y_b, psi])

        if numpy.abs(omega) > self.state_range[0,1]: # Bicycle fell over
            return -1.0, True
        elif self.isAtGoal():
            return self.reward_goal, True
        elif not self.navigate:
            return self.reward_shaping, False
        else:
            goal_angle = matrix.vector_angle(self.goal_loc, numpy.array([x_f-x_b, y_f-y_b])) * numpy.pi / 180.
            return (4. - goal_angle**2) * self.reward_shaping, False

    def isAtGoal(self):
        # Anywhere in the goal radius
        if self.navigate:
            return numpy.sqrt(max(0.,((self.position[:2] - self.goal_loc)**2).sum() - self.goal_rsqrd)) < 1.e-5
        else:
            return False

    def env_step(self,thisAction):
        intAction = thisAction.intArray[0]
        theReward, episodeOver = self.takeAction(intAction)

        theObs = Observation()
        theObs.doubleArray = self.state.tolist()
        returnRO = Reward_observation_terminal()
        returnRO.r = theReward
        returnRO.o = theObs
        returnRO.terminal = int(episodeOver)

        return returnRO

    def env_cleanup(self):
        pass

    def env_message(self,inMessage):
        return "I don't know how to respond to your message";



