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
from rlglue.environment import EnvironmentLoader as EnvironmentLoader
from pyrl.rlglue.registry import register_environment

from . import gridworld
from pyrl.misc.matrix import mvnpdf

@register_environment
class PuddleWorld(gridworld.Gridworld):
    name = "Puddle World"
    def __init__(self, size_x=10, size_y=10, goal_x=10, goal_y=10, puddle_penalty=-100.0,
             puddle_means=[(0.35, 0.5), (0.5, 0.35)], puddle_var=[(1.2, 1.e-5, 1.e-5, 0.5), (0.5, 1.e-5, 1.e-5, 1.2)],
             noise=0.0, reward_noise=0.0, random_start=False, fudge=1.4143):

        gridworld.Gridworld.__init__(self, size_x=size_x, size_y=size_y, goal_x=goal_x,
                         goal_y=goal_y, noise=noise, reward_noise=reward_noise, random_start=random_start, fudge=fudge)
        self.puddle_penalty = puddle_penalty
        self.puddle_means = map(numpy.array, puddle_means)
        self.puddle_var = map(lambda cov: numpy.linalg.inv(numpy.array(cov).reshape((2,2))), puddle_var)
        self.domain_name = "Continuous PuddleWorld"

    def reset(self):
        if self.random_start:
            self.pos = numpy.random.random((2,)) * self.size
        else:
            self.pos = numpy.array([0., 0.])

    def takeAction(self, action):
        base_reward = gridworld.Gridworld.takeAction(self, action)
        for mu, inv_cov in zip(self.puddle_means, self.puddle_var):
            base_reward += mvnpdf(self.pos, mu, inv_cov) * self.puddle_penalty
        return base_reward


if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Run Noisy Continuous Puddle World environment in network mode.')
    gridworld.addGridworldArgs(parser)
    parser.add_argument("--puddle", type=float, nargs=6, action='append',
                help="Add a puddle with arguments: mean_x, mean_y, cov1, cov2, cov3, cov4. " + \
                    "Where mean specifies the center of the puddle and cov specifies the " + \
                    "covariance matrix of the multivariate normal distribution that describes " + \
                    "the puddle's depth.")
    parser.add_argument("--puddle_penalty", type=float, default=-100,
                help="The reward penalty scale for walking through puddles.")
    args = parser.parse_args()
    kwargs = {}
    if args.puddle is not None:
        means = []
        covs = []
        for puddle in args.puddle:
            means.append(tuple(puddle[:2]))
            covs.append(tuple(puddle[2:]))
        kwargs['puddle_means'] = means
        kwargs['puddle_var'] = covs

    if args.size_x:
        kwargs['size_x'] = args.size_x
    if args.size_y:
        kwargs['size_y'] = args.size_y
    if args.goal_x:
        kwargs['goal_x'] = args.goal_x
    if args.goal_y:
        kwargs['goal_y'] = args.goal_y
    if args.noise:
        kwargs['noise'] = args.noise
    if args.fudge:
        kwargs['fudge'] = args.fudge
    if args.random_restarts:
        kwargs['random_start'] = args.random_restarts

    EnvironmentLoader.loadEnvironment(PuddleWorld(**kwargs))
