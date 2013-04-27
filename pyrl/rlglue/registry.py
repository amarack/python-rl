"""
.. module:: RLGlueRegistry
   :platform: Unix, Windows
   :synopsis: Registry for rl-glue agents, environments and experiments

.. moduleauthor:: Pierre-Luc Bacon <pierrelucbacon@gmail.com>

"""

class RLGlueRegistry(object):
    def __init__(self):
        self.agents = {}
        self.environments = {}
        self.experiments = {}

    def register_agent(self, cls):
        self.agents[cls.name] = cls
        return cls

    def register_environment(self, cls):
        self.environments[cls.name] = cls
        return cls

    def register_experiment(self, cls):
        self.experiments[cls.name] = cls
        return cls

rlglue_registry = RLGlueRegistry()
register_agent = rlglue_registry.register_agent
register_environment = rlglue_registry.register_environment
register_experiment = rlglue_registry.register_experiment
