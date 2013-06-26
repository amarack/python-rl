
# Author: Will Dabney

from rlglue.agent.Agent import Agent
from rlglue.agent import AgentLoader as AgentLoader
from rlglue.types import Action
from rlglue.types import Observation
from rlglue.utils import TaskSpecVRLGLUE3
from pyrl.rlglue.registry import register_agent

from random import Random
import numpy
import copy
import skeleton_agent

from pyrl.agents.models import batch_model
from pyrl.agents.planners import fitted_qiteration

@register_agent
class ModelBasedAgent(skeleton_agent.skeleton_agent):
    """
    ModelBasedAgent provides a reinforcement learning agent which plans, using the planner class provided,
    over a model of the domain, learned by the model learning class provided. So, essentially this class is
    just a wrapper around the real functionality of the planner and modeling classes.
        """

    name = "Model Based Agent"


    def init_parameters(self):
        model_class = self.params.setdefault('model_class', batch_model.KNNBatchModel)
        planner_class = self.params.setdefault('planner_class', fitted_qiteration.FittedQIteration)
        self.model = model_class(**(self.params.setdefault('model_params', {})))
        self.planner = planner_class(self.model, **(self.params.setdefault('planner_params', {})))

    def agent_supported(self, parsedSpec):
        if parsedSpec.valid:
            # Check observation form, and then set up number of features/states
            assert len(parsedSpec.getDoubleObservations()) + len(parsedSpec.getIntObservations()) > 0, "Expecting at least one continuous or discrete observation"

            # Check action form, and then set number of actions
            assert len(parsedSpec.getIntActions())==1, "Expecting 1-dimensional discrete actions"
            assert len(parsedSpec.getDoubleActions())==0, "Expecting no continuous actions"
            assert not parsedSpec.isSpecial(parsedSpec.getIntActions()[0][0]), "Expecting min action to be a number not a special value"
            assert not parsedSpec.isSpecial(parsedSpec.getIntActions()[0][1]), "Expecting max action to be a number not a special value"
            return True
        else:
            return False

    def agent_init(self,taskSpec):
        """Initialize the RL agent.

        Args:
            taskSpec: The RLGlue task specification string.
        """
        # (Re)initialize parameters (incase they have been changed during a trial
        self.init_parameters()
        # Parse the task specification and set up the weights and such
        TaskSpec = TaskSpecVRLGLUE3.TaskSpecParser(taskSpec)
        if self.agent_supported(TaskSpec):
            self.numStates=len(TaskSpec.getDoubleObservations())
            self.discStates = numpy.array(TaskSpec.getIntObservations())
            self.numDiscStates = int(reduce(lambda a, b: a * (b[1] - b[0] + 1), self.discStates, 1.0))
            self.numActions=TaskSpec.getIntActions()[0][1]+1;

            self.model.model_init(self.numDiscStates, TaskSpec.getDoubleObservations(), \
                              self.numActions, TaskSpec.getRewardRange()[0])
            self.planner.planner_init(self.numDiscStates, TaskSpec.getDoubleObservations(), \
                              self.numActions, TaskSpec.getRewardRange()[0])

        else:
            print "Task Spec could not be parsed: "+taskSpecString;

        self.lastAction=Action()
        self.lastObservation=Observation()


    def getAction(self, state, discState):
        """Get the action under the current plan policy for the given state.

        Args:
            state: The array of continuous state features
            discState: The integer representing the current discrete state value

        Returns:
            The current greedy action under the planned policy for the given state.
        """
        s = numpy.zeros((len(state) + 1,))
        s[0] = discState
        s[1:] = state
        a = self.planner.getAction(s)
        return a

    def getDiscState(self, state):
        """Return the integer value representing the current discrete state.

        Args:
            state: The array of integer state features

        Returns:
            The integer value representing the current discrete state
        """
        if self.numDiscStates > 1:
            x = numpy.zeros((self.numDiscStates,))
            mxs = self.discStates[:,1] - self.discStates[:,0] + 1
            mxs = numpy.array(list(mxs[:0:-1].cumprod()[::-1]) + [1])
            x = numpy.array(state) - self.discStates[:,0]
            return (x * mxs).sum()
        else:
            return 0

    def agent_start(self,observation):
        """Start an episode for the RL agent.

        Args:
            observation: The first observation of the episode. Should be an RLGlue Observation object.

        Returns:
            The first action the RL agent chooses to take, represented as an RLGlue Action object.
        """
        theState = numpy.array(list(observation.doubleArray))
        thisIntAction=self.getAction(theState, self.getDiscState(observation.intArray))
        returnAction=Action()
        returnAction.intArray=[thisIntAction]

        self.lastAction=copy.deepcopy(returnAction)
        self.lastObservation=copy.deepcopy(observation)

        return returnAction

    def agent_step(self,reward, observation):
        """Take one step in an episode for the agent, as the result of taking the last action.

        Args:
            reward: The reward received for taking the last action from the previous state.
            observation: The next observation of the episode, which is the consequence of taking the previous action.

        Returns:
            The next action the RL agent chooses to take, represented as an RLGlue Action object.
        """
        newState = numpy.array(list(observation.doubleArray))
        lastState = numpy.array(list(self.lastObservation.doubleArray))
        lastAction = self.lastAction.intArray[0]

        newDiscState = self.getDiscState(observation.intArray)
        lastDiscState = self.getDiscState(self.lastObservation.intArray)

        phi_t = numpy.zeros((self.numStates+1,))
        phi_tp = numpy.zeros((self.numStates+1,))
        phi_t[0] = lastDiscState
        phi_t[1:] = lastState
        phi_tp[0] = newDiscState
        phi_tp[1:] = newState

        #print ','.join(map(str, lastState))

        self.planner.updateExperience(phi_t, lastAction, phi_tp, reward)

        newIntAction = self.getAction(newState, newDiscState)
        returnAction=Action()
        returnAction.intArray=[newIntAction]

        self.lastAction=copy.deepcopy(returnAction)
        self.lastObservation=copy.deepcopy(observation)
        return returnAction

    def agent_end(self,reward):
        """Receive the final reward in an episode, also signaling the end of the episode.

        Args:
            reward: The reward received for taking the last action from the previous state.
        """
        lastState = numpy.array(list(self.lastObservation.doubleArray))
        lastAction = self.lastAction.intArray[0]
        lastDiscState = self.getDiscState(self.lastObservation.intArray)

        phi_t = numpy.zeros((self.numStates+1,))
        phi_t[0] = lastDiscState
        phi_t[1:] = lastState

        self.planner.updateExperience(phi_t, lastAction, None, reward)

    def agent_cleanup(self):
        """Perform any clean up operations before the end of an experiment."""
        pass

    def has_diverged(self, values):
        value = values.sum()
        return numpy.isnan(value) or numpy.isinf(value)

    def agent_message(self,inMessage):
        """Receive a message from the environment or experiment and respond.

        Args:
            inMessage: A string message sent by either the environment or experiment to the agent.

        Returns:
            A string response message.
        """
        if inMessage.lower() == "agent_diverged?": # If we find that this is needed, we can fill it in later
            return "False" #str(self.has_diverged(self.weights))
        else:
            return name + " does not understand your message."


if __name__=="__main__":
    from pyrl.agents.skeleton_agent import runAgent
    runAgent(ModelBasedAgent)

# If executed as a standalone script this will default to RLGlue network mode.
# Some parameters can be passed at the command line to customize behavior.
# if __name__=="__main__":
#     import argparse
#     parser = argparse.ArgumentParser(description='Run ModelBasedAgent in network mode')
#     parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
#     parser.add_argument("--model", type=str, default="knn", help="What model class to use", choices=["knn", "randforest", "svm", "gp"])
#     parser.add_argument("--planner", type=str, default="fittedq", help="What planner class to use", choices=["fittedq"])
#     parser.add_argument("--svmde",  action='store_true', help="Use the one class SVM density estimator for known/unknown distinctions.")
#     args = parser.parse_args()

#     model_params = {}
#     planner_params = {}
#     model_class = None
#     planner_class = None

#     if args.model == "knn":
#         model_params = {"update_freq": 20, "known_threshold": 0.95, "max_experiences": 700}
#         if args.svmde:
#             model_class = batch_model.KNNSVM
#         else:
#             model_class = batch_model.KNNBatchModel
#     elif args.model == "randforest":
#         model_params = {"known_threshold": 0.95, "max_experiences": 800, "importance_weight": True}
#         if args.svmde:
#             model_class = model_class = batch_model.RandForestSVM
#         else:
#             model_class = batch_model.RandomForestBatchModel
#     elif args.model == "svm":
#         model_params = {"known_threshold": 0.95, "max_experiences": 500, "importance_weight": True}
#         if args.svmde:
#             model_class = batch_model.SVM2
#         else:
#             model_class = batch_model.SVMBatchModel
#     elif args.model == "gp":
#         model_params = {"max_experiences": 300, "nugget": 1.0e-10, "random_start": 100}
#         if args.svmde:
#             model_class = batch_model.GPSVM
#         else:
#             model_class = batch_model.GaussianProcessBatchModel

#     if args.planner == "fitqit":
#         planner_params = {"basis": "fourier", "regressor": "ridge", "iterations": 1000, "support_size": 50, "resample": 15}
#         planner_class = fitted_qiteration.FittedQIteration

#     params = {'gamma': args.gamma, 'model_class': model_class, 'model_params': model_params,
#           'planner_class': planner_class, 'planner_params': planner_params}

#     AgentLoader.loadAgent(ModelBasedAgent(params))

