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
import copy
from rlglue.agent.Agent import Agent
from rlglue.agent import AgentLoader as AgentLoader
from rlglue.types import Action
from rlglue.types import Observation
from rlglue.utils import TaskSpecVRLGLUE3

from pyrl.rlglue.registry import register_agent


@register_agent
class skeleton_agent(Agent):
    name = "Skeleton agent"

    def __init__(self, **args):
        self.randGenerator = Random()
        lastAction = Action()
        lastObservation = Observation()

    def randomize_parameters(self, **args):
        """Generate parameters randomly, constrained by given named parameters.

        Args:
            **args: Named parameters to fix, which will not be randomly generated

        Returns:
            List of resulting parameters of the class. Will always be in the same order.
            Empty list if parameter free.

        """
        return []

    def agent_init(self,taskSpec):
        """Initialize the RL agent.

        Args:
            taskSpec: The RLGlue task specification string.
        """
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
        return "Agent does not understand your message."


if __name__=="__main__":
    AgentLoader.loadAgent(skeleton_agent())
