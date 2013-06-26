#
# A random agent, and very simple example of the core methods needed for
# an RL-Glue/pyRL agent. The RL-Glue methods all begin with "agent_",
# and pyRL expects __init__ and randomize_parameters of the form shown
# here.
#
# This was based upon the skeleton_agent that comes with the RL-Glue
# python codec (Brian Tanner, 2008), but now it really is just the simplest
# random agent that fits into this framework.

from random import Random
import copy, numpy
from rlglue.agent.Agent import Agent
from rlglue.agent import AgentLoader as AgentLoader
from rlglue.types import Action
from rlglue.types import Observation
from rlglue.utils import TaskSpecVRLGLUE3

from pyrl.rlglue.registry import register_agent
from pyrl.misc.parameter import *

@register_agent
class skeleton_agent(Agent, object):
    name = "Skeleton agent"

    def __init__(self, **args):
        self.randGenerator = Random()
        lastAction = Action()
        lastObservation = Observation()
        self.params = args
        self.init_parameters()

    def init_parameters(self):
        """Initialize algorithm parameters. Will be called by constructor, and at the
        start of each new run. Parameters' initial values should be stored in
        self.params, and here instances of them should be copied into object variables
        which may or may not change during a particular run of the algorithm.
        """
        pass

    @classmethod
    def agent_parameters(cls):
        """Produces an argparse.ArgumentParser for all the parameters of this RL agent
        algorithm. Specifically, parameters mean to be optimized (e.g. in a parameter search)
        should be added to the argument group 'optimizable'. The best way to do this is with
        the functions contained in pyrl/misc/parameter.py. Specifically, parameter_set for
        creating a new set of parameters, and add_parameter to add parameters (use optimize=False)
        to indicate that the parameter should not be optimized over.
        """
        return parameter_set(cls.name, description="Parameters required for running an RL agent algorithm.")

    def agent_init(self,taskSpec):
        """Initialize the RL agent.

        Args:
            taskSpec: The RLGlue task specification string.
        """
        self.init_parameters()
        # Consider looking at sarsa_lambda agent for a good example of filling out these methods
        TaskSpec = TaskSpecVRLGLUE3.TaskSpecParser(taskSpec)
        assert len(TaskSpec.getIntActions())==1
        self.numActions = TaskSpec.getIntActions()[0][1]+1

        self.lastAction = Action()
        self.lastObservation = Observation()
        self.counter = 0

    def agent_start(self,observation):
        """Start an episode for the RL agent.

        Args:
            observation: The first observation of the episode. Should be an RLGlue Observation object.

        Returns:
            The first action the RL agent chooses to take, represented as an RLGlue Action object.
        """

        # Generate a random action
        thisIntAction = self.randGenerator.randint(0,self.numActions-1)
        returnAction = Action()
        returnAction.intArray = [thisIntAction]

        lastAction = copy.deepcopy(returnAction)
        lastObservation = copy.deepcopy(observation)

        return returnAction

    def agent_step(self,reward, observation):
        """Take one step in an episode for the agent, as the result of taking the last action.

        Args:
            reward: The reward received for taking the last action from the previous state.
            observation: The next observation of the episode, which is the consequence of taking the previous action.

        Returns:
            The next action the RL agent chooses to take, represented as an RLGlue Action object.
        """

        # Generate a random action
        thisIntAction = self.randGenerator.randint(0,self.numActions-1)
        returnAction = Action()
        returnAction.intArray = [thisIntAction]

        lastAction = copy.deepcopy(returnAction)
        lastObservation = copy.deepcopy(observation)

        return returnAction

    def agent_end(self,reward):
        """Receive the final reward in an episode, also signaling the end of the episode.

        Args:
            reward: The reward received for taking the last action from the previous state.
        """
        pass

    def agent_cleanup(self):
        """Perform any clean up operations before the end of an experiment."""
        pass

    def agent_message(self,inMessage):
        """Receive a message from the environment or experiment and respond.

        Args:
            inMessage: A string message sent by either the environment or experiment to the agent.

        Returns:
            A string response message.
        """
        if inMessage.lower() == "agent_diverged?":
            return str(self.has_diverged())
        else:
            return name + " does not understand your message."

    def has_diverged(self):
        """Overwrite the function with one that checks the key values for your
        agent, and returns True if they have diverged (gone to nan or infty for ex.), and
        returns False otherwise.
        """

        return False

def runAgent(agent_class):
    """Use the agent_parameters function to parse command line arguments
    and run the RL agent in network mode.
    """
    parser = argparse.ArgumentParser(parents=[agent_class.agent_parameters()], add_help=True)
    params = vars(parser.parse_args())
    AgentLoader.loadAgent(agent_class(**params))


if __name__=="__main__":
    runAgent(skeleton_agent)

