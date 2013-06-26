

import numpy
import qlearning
from pyrl.rlglue.registry import register_agent

@register_agent
class delayed_qlearning(qlearning.qlearning_agent):
    """Delayed Q-Learning algorithm. This algorithm is only directly applicable
    to discrete state, discrete action domains. Thus, it should throw an assertion
    failure if you attempt to use it not in such a domain.

    Unfortunately, I have no yet been able to get this to work consistently on
    the marble maze domain. It seems likely that it would work on something simpler
    like chain domain. Maybe there's a bug?

    From the paper:
    PAC Model-Free Reinforcement Learning. 2006.
    Alexander Strehl, Lihong Li, Eric Wiewiora, John Langford, and Michael Littman.
    """

    name = "Delayed Q-Learning"

    def init_parameters(self):
        self.gamma = self.params.setdefault('gamma', 0.99)
        self.epsilon = self.params.setdefault('epsilon', 0.1)
        super(delayed_qlearning, self).init_parameters()
        self.m = self.params.setdefault('m', 100)

    @classmethod
    def agent_parameters(cls):
        param_set = parameter_set(cls.name, description="Parameters required for running an RL agent algorithm.")
        add_parameter(param_set, "epsilon", default=0.1)
        add_parameter(param_set, "gamma", default=0.99)
        add_parameter(param_set, "m", default=100, type=int, min=1, max=1000)
        return param_set

    def agent_supported(self, parsedSpec):
        if parsedSpec.valid:
            # Check observation form, and then set up number of features/states
            assert len(parsedSpec.getIntObservations()) > 0, "Expecting at least one discrete observation"
            assert len(parsedSpec.getDoubleObservations()) == 0, "Expecting no continuous observations."

            # Check action form, and then set number of actions
            assert len(parsedSpec.getIntActions())==1, "Expecting 1-dimensional discrete actions"
            assert len(parsedSpec.getDoubleActions())==0, "Expecting no continuous actions"
            assert not parsedSpec.isSpecial(parsedSpec.getIntActions()[0][0]), "Expecting min action to be a number not a special value"
            assert not parsedSpec.isSpecial(parsedSpec.getIntActions()[0][1]), "Expecting max action to be a number not a special value"
            self.reward_range = numpy.array(parsedSpec.getRewardRange()[0])
            return True
        else:
            return False

    def agent_init(self,taskSpec):
        super(delayed_qlearning, self).agent_init(taskSpec)
        self.weights.fill(1./(1. - self.gamma))
        self.updates = numpy.zeros(self.weights.shape)
        self.visit_count = numpy.zeros(self.weights.shape)
        self.update_time = numpy.zeros(self.weights.shape)
        self.LEARN = numpy.ones(self.weights.shape, dtype=bool)
        self.last_update = 0
        self.step_count = 0
        # Compute the 'correct' m to use (from the paper)
        # But tends to be so large as to be impractical
        #k = 1./((1. - self.gamma)*self.epsilon)
        #delta = 0.1
        #self.m = numpy.log(3. * self.numDiscStates * self.numActions * (1. + self.numDiscStates * self.numActions * k) / delta)
        #self.m /= 2. * self.epsilon**2 * (1. - self.gamma)**2
        #self.m = int(self.m)
        #print self.m

    def getAction(self, state, discState):
        """Get the action under the current policy for the given state.

        Args:
            state: The array of continuous state features
            discState: The integer representing the current discrete state value

        Returns:
            The current policy action, or a random action with some probability.
        """
        return numpy.dot(self.weights[discState,:,:].T, self.basis.computeFeatures(state)).argmax()

    def update(self, phi_t, state, discState, reward):
        reward = (reward - self.reward_range[0]) / (self.reward_range[1] - self.reward_range[0])
        self.step_count += 1
        state_action = numpy.where(phi_t != 0)
        if self.LEARN[state_action]: # If Learn[s,a]
            qvalues = self.getActionValues(state, discState)
            self.updates[state_action] += reward + self.gamma * qvalues.max()
            self.visit_count[state_action] += 1
            if self.visit_count[state_action] == self.m:
                if self.weights[state_action] - self.updates[state_action]/self.m >= 2. * self.epsilon:
                    self.weights[state_action] = self.updates[state_action]/self.m + self.epsilon
                    self.last_update = self.step_count
                    #print (self.weights.ravel() < self.weights.max()).sum(), self.weights.size
                elif self.update_time[state_action] >= self.last_update:
                    self.LEARN[state_action] = False
                self.update_time[state_action] = self.step_count
                self.updates[state_action] = 0
                self.visit_count[state_action] = 0
        elif self.update_time[state_action] < self.last_update:
            self.LEARN[state_action] = True

if __name__=="__main__":
    from pyrl.agents.skeleton_agent import runAgent
    runAgent(delayed_qlearning)



