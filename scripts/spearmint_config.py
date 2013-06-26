
# Author: Will Dabney

#import csv, os, json, numpy, copy
import numpy, sys
sys.path.append("..")
from pyrl.misc.parameter import *

def gen_numeric_pbvar(name, size, type, min, max):
    config = ["variable {", ' name: "' + name + '"']
    if type is int:
        config.append(" type: INT")
    else:
        config.append(" type: FLOAT")
    config.append(" size: " + str(size))
    config.append(" min: " + str(min))
    config.append(" max: " + str(max))
    config += ["}", ""]
    return config

def gen_enum_pbvar(name, size, options):
    config = ["variable {", ' name: "' + name + '"']
    config.append(" type: ENUM")
    config.append(" size: " + str(size))
    for entry in options:
        config.append(' options: "' + str(entry) + '"')
    config += ["}", ""]
    return config

def gen_config(agent_name, param_parser, fixed_params):
    config_contents = ["language: PYTHON", 'name: "' + agent_name + '"', ""]

    opt_grp = get_optimize_group(param_parser)
    opt_pnames = set(fixed_params.keys())
    for param in opt_grp._group_actions:
        if (param.dest in opt_pnames):
            continue
        opt_pnames.add(param.dest)

        var_size = param.nargs if param.nargs is not None else 1
        if param.type is bool:
            config_contents += gen_enum_pbvar(param.dest, var_size, ["true", "false"])
        elif param.choices.__class__ is ValueRange:
            config_contents += gen_numeric_pbvar(param.dest, var_size, param.type,
                param.choices.min(), param.choices.max())
        else:
            config_contents += gen_enum_pbvar(param.dest, var_size, param.choices)

    return config_contents

if __name__ == "__main__":
    from pyrl.agents import *
    from pyrl.rlglue.registry import rlglue_registry
    from pyrl.rlglue import run

    agent_name = sys.argv[1]
    agent = rlglue_registry.agents[agent_name]
    param_parser = agent.agent_parameters()
    fixed_params = run.fromjson(sys.argv[2])[1] # Grabs agent parameters from experiment file

    # Produce a config.pb file based upon parameter parser
    config_contents = gen_config(agent_name, param_parser, fixed_params)
    for line in config_contents:
        print line








