from pyrl.environments.skeleton_environment import skeleton_environment
from pyrl.agents.skeleton_agent import skeleton_agent
import pyrl.experiments.episodic as episodic

episodic.setupExperiment(1000,skeleton_environment(), skeleton_agent())
episodic.runTrial(10, "test.dat", True)
