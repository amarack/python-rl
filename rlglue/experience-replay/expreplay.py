
import cPickle

from rlglue.types import Action
from rlglue.types import Observation
from rlglue.utils import TaskSpecVRLGLUE3


class experience_saver:
    def __init__(self, filename, bufsize=1000):
        self.filename = filename
        self.buffer = []
        self.task_spec = None
        self.bufsize = bufsize

    def setTaskSpecString(self, taskSpecString):
        self.task_spec = taskSpecString
        self.buffer.append(taskSpecString)

    def addStep(self, observation, action, reward):
        self.buffer.append([observation, action, reward])
        if len(self.buffer) > self.bufsize:
            self.flush()

    def startEpisode(self, observation, action):
        self.buffer.append([observation, action])

    def endEpisode(self, reward):
        self.buffer.append([reward])

    def flush(self):
        file = open(self.filename, "a")
        for line in self.buffer:
            cPickle.dump(line ,file)
        file.close()
        self.buffer = []

class experience_replay:
    def __init__(self, filename, agent):
        self.input_file = open(filename, "r")
        taskSpecString = cPickle.load(self.input_file)
        agent.agent_init(taskSpecString)
        while True:
            try:
                step = cPickle.load(self.input_file)
                if len(step) == 2:
                    agent.offpolicy_start(step[0], step[1])
                elif len(step) == 3:
                    agent.offpolicy_step(step[0], step[1], step[2])
                else:
                    agent.agent_end(step[0])
            except EOFError:
                break
