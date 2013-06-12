
# Author: Will Dabney
# Author: Pierre-Luc Bacon <pierrelucbacon@gmail.com>

# Runs an experiment by starting up rl_glue,
# and letting the user choose from a set of
# agents, environments, and experiments.

import json
from multiprocessing import Process
from subprocess import Popen

from pyrl.agents import *
from pyrl.environments import *
from pyrl.experiments import *
from pyrl.rlglue.registry import rlglue_registry
from pyrl.misc.json import convert

from rlglue.agent import AgentLoader as AgentLoader
from rlglue.environment import EnvironmentLoader as EnvironmentLoader


def fromjson(filename):
    with open(filename, 'r') as f:
        config = json.load(f, object_hook=convert)

    # Process the environment
    environment = rlglue_registry.environments[config['environment']['name']]
    environment_params = config['environment']['params']
    # Process the agent
    agent = rlglue_registry.agents[config['agent']['name']]
    agent_params = config['agent']['params']
    # Process the experiment
    experiment = rlglue_registry.experiments[config['experiment']['name']]
    experiment_params = config['experiment']['params']

    return agent, agent_params, environment, environment_params, experiment, experiment_params

def tojson(agent, a_args, env, env_args, exp, exp_args, local=None):
    config = {'agent': {'name': agent.name, 'params': a_args},
              'environment': {'name': env.name, 'params': env_args},
              'experiment': {'name': exp.name, 'params': exp_args}}
    return json.dumps(config)

def fromuser():
    environment = interactive_choose(rlglue_registry.environments,
                                     "Choose an environment.")
    agent = interactive_choose(rlglue_registry.agents, "Choose an agent.")
    experiment = interactive_choose(rlglue_registry.experiments,
                                    "Choose an experiment.")
    return agent, {}, environment, {}, experiment, {}


def interactive_choose(choices, prompt):
    print(prompt)
    sortkeys = sorted(choices.keys())

    for ix, a_key in enumerate(sortkeys):
        print("  ({:d}): {}".format(ix + 1, a_key))

    choice = None
    while choice not in range(1, len(sortkeys) + 1):
        choice = raw_input("Enter number (1 - {:d}): ".format(
            len(sortkeys)))
        try:
            choice = int(choice)
        except:
            pass

    return choices[sortkeys[choice - 1]]


def run(agent, a_args, env, env_args, exp, exp_args, local=None, result_file=None):
    if local is None:
        ans = raw_input("Run locally? [y/n]: ")
        if ans.lower() == 'y' or ans.lower() == 'yes':
            local = True
        else:
            local = False

    config = {'agent': {'name': agent.name, 'params': a_args},
              'environment': {'name': env.name, 'params': env_args},
              'experiment': {'name': exp.name, 'params': exp_args}}
    if local:
        experiment = exp(config, agent=agent(**a_args),
                         environment=env(**env_args), **exp_args)
        experiment.run_experiment(filename=result_file)
    else:
        experiment = exp(config, **exp_args)
        # TODO: Figure out if rl_glue is running, don't start it in that case
        rlglue_p = Popen('rl_glue')
        agent_p = Process(target=AgentLoader.loadAgent,
                          args=(agent(**a_args),))
        agent_p.start()
        env_p = Process(target=EnvironmentLoader.loadEnvironment,
                        args=(env(**env_args),))
        env_p.start()
        experiment.run_experiment(filename=result_file, **a_args)
        env_p.terminate()
        agent_p.terminate()
        rlglue_p.terminate()


def addRunExpArgs(parser):
    json_group = parser.add_mutually_exclusive_group()
    json_group.add_argument("--load", type=str, help="Load an experimental configuration from a JSON file.")
    json_group.add_argument("--genjson", action='store_true', help="Generate an experimental configuration JSON file from " + \
                            "interactive selections. Only generates, does not run.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--local", action='store_true', default="True", help="Run experiment locally")
    group.add_argument("--network", action='store_true', help="Run experiment through network sockets")
    parser.add_argument("--output", type=str, help="Save the results to a file.")
    return parser

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run a reinforcement learning experiment. Defaults to interactive experiment.')
    addRunExpArgs(parser)
    args = parser.parse_args()

    if args.load is None:
        config = fromuser()
        if args.genjson:
            print tojson(*config)
        else:
            run(*config,local=args.local, result_file=args.output)
    else:
        config = fromjson(args.load)
        run(*config, local=args.local, result_file=args.output)
