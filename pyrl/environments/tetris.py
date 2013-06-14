#
# Copyright (C) 2013, Will Dabney
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#

import os, glob, sys, numpy

from rlglue.environment.Environment import Environment
from rlglue.environment import EnvironmentLoader as EnvironmentLoader
from rlglue.types import Observation
from rlglue.types import Action
from rlglue.types import Reward_observation_terminal

from pyrl.rlglue import TaskSpecRLGlue
from pyrl.rlglue.registry import register_environment

try:
    from mdptetris import mdptetris
except ImportError, e:
    print 'mdptetris not available ', e
    print "Please build mdptetris first by running 'make' from pyrl/"

@register_environment
class Tetris(Environment):
    name = "Tetris"

    def __init__(self, original_features=True, dellacherie_features=False, width=10, height=20,
				pieces_filename=os.path.join(os.path.dirname(__file__),
                                             'configs', 'tetris', 'standard.dat')):
        """This class provides the Tetris domain.

        This uses our port of the mdptetris project from:
        https://gforge.inria.fr/projects/mdptetris/
        http://mdptetris.gforge.inria.fr/doc/

        Allows agents to play the game of tetris in any of the varieties
        implemented. The default is a simplified tetris game in which the
        agent sets the rotation/orientation and column to drop each piece, but
        does not control the piece during its descent.

        This allows loading of the pieces used in the game from a file.
        Example files for standard tetris configurations are found in
        pyrl/environments/configs/tetris/

        Many features used in the literature are implemented, and can be
        chosen for inclusion in the observation.
        """

        self.original_features = original_features
        self.dellacherie_features = dellacherie_features
        self.board_width = width
        self.board_height = height
        self.pieces_filename = pieces_filename
        self.domain_name = "Tetris"

    # The reward function is not particularly standardized in Tetris
    # So, we make it easy to override, but default to something reasonable.
    def computeRewardRange(self):
        # Reward: 1 per step (piece placed), 1 per line cleared, 0 for gameover
        # The maximum would be: 1 + max_piece_height, for now we will assume standard pieces
        return (0, 5)

    def computeReward(self, lines_cleared):
        return lines_cleared

    # For our default behavior the actions are 2 dimensional,
    # but we map it onto a one dimensional discrete action
    def maxAction(self):
        return self.board_width * 4

    def printBoard(self):
        mdptetris.game_print()

    def makeTaskSpec(self):
        ts = TaskSpecRLGlue.TaskSpec(discount_factor=1.0, reward_range=self.computeRewardRange())
        ts.addDiscreteAction((0.0, self.maxAction()-1))
        ts.addDiscreteObservation((0, mdptetris.num_pieces() - 1))
        ts.addContinuousObservation((1.0,1.0))
        if self.original_features:
            ranges = mdptetris.feature_ranges("original")
            for rng in ranges:
                ts.addContinuousObservation(rng)
        if self.dellacherie_features:
            ranges = mdptetris.feature_ranges("dellacherie")
            for rng in ranges:
                ts.addContinuousObservation(rng)
        ts.setEpisodic()
        ts.setExtra(self.domain_name)
        return ts.toTaskSpec()

    def reset(self):
        mdptetris.reset_game()

    def getObservation(self):
        returnObs = Observation()
        features = [1.]
        if self.original_features:
            features += mdptetris.features_original()
        if self.dellacherie_features:
            features += mdptetris.features_dellacherie()

        returnObs.intArray = [mdptetris.current_piece()]
        returnObs.doubleArray = features
        return returnObs

    def env_init(self):
        # Start a game of tetris
        mdptetris.tetris(self.board_width, self.board_height, 0, self.pieces_filename)
        return self.makeTaskSpec()

    def env_start(self):
        self.reset()
        returnObs = self.getObservation()
        return returnObs

    def takeAction(self, intAction):
        # intAction is interpreted as an index into the
        # cross product between columns and rotations.
        rotation, column = intAction % 4, int(intAction/4)+1

        # Next, the game restricts the rotations and columns based
        # on the current piece. So, we map the selected rotation and column appropriately
        rotation %= mdptetris.num_rotate_actions()
        column = min(column, mdptetris.num_column_actions(rotation))

        # Take the action
        lines_cleared = mdptetris.drop_piece(rotation, column)
        #if lines_cleared > 0:
        #    print "Cleared!"
        #print "Action", rotation, column
        obs = self.getObservation()
        reward = self.computeReward(lines_cleared) if not mdptetris.isgameover() else -1.0
        return obs, reward

    def env_step(self,thisAction):
        intAction = thisAction.intArray[0]
        obs, reward = self.takeAction(intAction)

        theObs = obs

        returnRO = Reward_observation_terminal()
        returnRO.r = reward
        returnRO.o = theObs
        returnRO.terminal = mdptetris.isgameover()

        return returnRO

    def env_cleanup(self):
        pass

    def env_message(self,inMessage):
        return "I don't know how to respond to your message";

