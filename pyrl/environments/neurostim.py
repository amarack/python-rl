
import numpy, os

from sklearn.neighbors import NearestNeighbors

from rlglue.environment.Environment import Environment
from rlglue.environment import EnvironmentLoader as EnvironmentLoader
from rlglue.types import Observation
from rlglue.types import Action
from rlglue.types import Reward_observation_terminal
from pyrl.rlglue import TaskSpecRLGlue
from pyrl.rlglue.registry import register_environment

@register_environment
class NeuroStim(Environment):
    name = "Neurostimulation"

    def __init__(self, filename=os.path.join(os.path.dirname(__file__),'configs', 'neurostim', 'params.dat'),
                noise=0.00001, stim_penalty=-1.0, seizure_penalty=-40.0):

        self.noise = noise
        self.stim_penalty = stim_penalty
        self.seizure_penalty = seizure_penalty
        self.embed = -8

        # Usefull utility function
        def samedir(file):
            return os.path.join(os.path.split(filename)[0], file)

        # Load model parameters from input file
        with open(filename, "rb") as f:
            lines = [line.rstrip() for line in f]

        self.features_filename = samedir(lines[0])
        self.labels_filename = samedir(lines[1])
        self.stimulation_filename = samedir(lines[2])
        self.N, self.E, self.L, self.seiz_label = map(int, lines[3:7])
        self.stim_magnitude = float(lines[7])
        self.Ndt = int(lines[8])

        # Load Data from Files...
        # Model features
        self.features = numpy.genfromtxt(self.features_filename)
        # Model labels
        self.labels = numpy.genfromtxt(self.labels_filename)
        # Prototype stimulation
        self.dstim = numpy.genfromtxt(self.stimulation_filename)

        # Initialize model state to zero
        self.state = numpy.zeros((self.E,))

        # Construct the Noise Model
        self.noise_mags = numpy.abs(self.features[1:,:] - self.features[:-1,:]).mean(0)

        # Construct K-d Tree (for Nearest Neighbor computation)
        self.knn = NearestNeighbors(n_neighbors=1, algorithm='kd_tree')
        self.knn.fit(self.features)

        # Initialize state
        self.reset()

    def makeTaskSpec(self):
        ts = TaskSpecRLGlue.TaskSpec(discount_factor=1.0, reward_range=(-41.0, 0.0))
        ts.setDiscountFactor(1.0)
        ts.addDiscreteAction((0, 1))
        for dim in range(self.features.shape[1]):
            ts.addContinuousObservation((self.getMinByDimension(dim), self.getMaxByDimension(dim)))

        ts.setContinuing()
        ts.setExtra(self.name)
        print ts.toTaskSpec()
        return ts.toTaskSpec()

    def env_init(self):
        return self.makeTaskSpec()

    def env_start(self):
        returnObs = Observation()
        returnObs.doubleArray = self.state.tolist()
        return returnObs

    def env_step(self,thisAction):
        episodeOver = 0
        theReward = -1.0
        intAction = thisAction.intArray[0]

        self.step(intAction, self.noise)
        seized = 0
        theReward = self.stim_penalty if intAction == 1 else 0.0
        if self.getLabel(self.current_neighbor) == self.seiz_label:
            theReward += self.seizure_penalty

        theObs = Observation()
        theObs.doubleArray = self.state.tolist()

        returnRO = Reward_observation_terminal()
        returnRO.r = theReward
        returnRO.o = theObs
        returnRO.terminal = 0

        return returnRO

    def env_cleanup(self):
        pass

    def env_message(self,inMessage):
        return "I don't know how to respond to your message";

    def reset(self, index=None):
        random_index = tuple(numpy.random.randint(self.N+1, size=2))
        self.state = self.features[random_index, :].mean(0) if index is None else self.features[index, :]

        # Compute nbr of state
        self.current_neighbor = self.knn.kneighbors(self.state)[1][0,0]

    def resetToState(self, state):
        self.state = state.copy()
        self.current_neighbor = self.knn.kneighbors(self.state)[1][0,0]

    def getMinByDimension(self, dim):
        return self.features[:,dim].min()

    def getMaxByDimension(self, dim):
        return self.features[:,dim].max()

    def getSize(self):
        return self.features.shape[0]

    def getDimensions(self):
        return self.features.shape[1]

    def step(self, stim, noise):
        # Check KNN bounds
        if self.current_neighbor >= self.features.shape[0]:
            self.reset()

        next_nbr = (self.current_neighbor + 1) % self.features.shape[0]

        # Compute the gradient
        gradient = self.features[next_nbr, :] - self.features[self.current_neighbor, :]

        # Compute the noise
        s_noise = noise * self.noise_mags * numpy.random.random((self.E,)) * 2.

        # Integrate
        self.state += gradient + s_noise + stim*self.stim_magnitude*self.dstim
        # Update KNN
        self.current_neighbor = self.knn.kneighbors(self.state)[1][0,0]

    def getLabel(self, nbr):
        return self.labels[nbr,2]



