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

    def randParameter(self, param_key, args, sample=numpy.random.random()):
        """A utility function for use inside randomize_parameters. Takes a parameter
        key (name), the named arguments passed to randomize_parameters, and optionally
        the sampled random value to set in case the key does not exist in the arguments.

        This will then set it (if not already present) in args and assign which ever value
        args ends up with into params.
        """
        self.params[param_key] = args.setdefault(param_key, sample)

    def randomize_parameters(self, **args):
        """Generate parameters randomly, constrained by given named parameters.

        Args:
            **args: Named parameters to fix, which will not be randomly generated

        Returns:
            List of resulting parameters of the class. Will always be in the same order.
            Empty list if parameter free.

        """
        return args

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



if __name__=="__main__":
    AgentLoader.loadAgent(skeleton_agent())
