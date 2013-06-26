
# Author: Will Dabney

import csv, os, json, numpy, copy
from pyrl.misc.timer import Timer
from pyrl.rlglue import RLGlueLocal as RLGlueLocal
from pyrl.rlglue.registry import register_experiment
import rlglue.RLGlue as rl_glue
from pyrl.experiments.episodic import Episodic
import pyrl.visualizers.plotExperiment as plotExperiment
from pyrl.misc.parameter import *


@register_experiment
class SpearmintGenerator(Episodic):
    name = "Spearmint Generator"

    def __init__(self, config, **kwargs):
        if not kwargs.has_key('agent') or not kwargs.has_key('environment'):
            print "ERROR: Spearmint Generator must be run locally in order to optimize parameters."
            import sys
            sys.exit(1)
        Episodic.__init__(self, config, **kwargs)

    def _gen_numeric_pbvar(self, name, size, type, min, max):
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

    def _gen_enum_pbvar(self, name, size, options):
        config = ["variable {", ' name: "' + name + '"']
        config.append(" type: ENUM")
        config.append(" size: " + str(size))
        for entry in options:
            config.append(' options: "' + str(entry) + '"')
        config += ["}", ""]
        return config

    def _gen_config(self, param_parser, fixed_params):
        config_contents = ["language: PYTHON", 'name: "' + self.agent.name + '"', ""]

        opt_grp = get_optimize_group(param_parser)
        opt_pnames = set(fixed_params.keys())
        for param in opt_grp._group_actions:
            if (param.dest in opt_pnames):
                continue
            opt_pnames.add(param.dest)

            var_size = param.nargs if param.nargs is not None else 1
            if param.type is bool:
                config_contents += self._gen_enum_pbvar(param.dest, var_size, ["true", "false"])
            elif param.choices.__class__ is ValueRange:
                config_contents += self._gen_numeric_pbvar(param.dest, var_size, param.type,
                                        param.choices.min(), param.choices.max())
            else:
                config_contents += self._gen_enum_pbvar(param.dest, var_size, param.choices)

        #for param in param_parser._actions:
        #    if (param.dest in opt_pnames) or (param.default is None):
        #        continue
        #    opt_pnames.add(param.dest)
        #    var_size = param.nargs if param.nargs is not None else 1
        #    config_contents += self._gen_enum_pbvar(param.dest, var_size, [param.default])

        #for key in fixed_params.keys():
        #    config_contents += self._gen_enum_pbvar(key, 1, [fixed_params[key]])
        return config_contents

    def run_experiment(self, filename=None):
        param_parser = self.agent.agent_get_parameters()
        fixed_params = self.configuration['agent']['params']

        # Produce a config.pb file based upon parameter parser
        config_contents = self._gen_config(param_parser, fixed_params)

        if filename is not None:
            # Get the absolute path to pyRL directory
            base_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../"))
            template_path = os.path.join(base_path, "pyrl", "experiments", "spearmint_run.py")
            experiment_path = os.path.join(filename, self.agent.name)
            os.system("mkdir " + filename)
            os.system("mkdir " + experiment_path)
            base_path = base_path.replace("/", "\/")
            os.system("cat " + template_path + " | " + \
                "sed -e 's/pyrl_path = \"###\"/pyrl_path = \"" + base_path + "\"/g' " + \
                "> " + os.path.join(experiment_path, self.agent.name + ".py"))

            with open(os.path.join(experiment_path, "experiment.json"), "w") as f:
                f.write(json.dumps(self.configuration) + "\n")

            with open(os.path.join(experiment_path, "config.pb"), "w") as f:
                f.writelines([k + "\n" for k in config_contents])
        else:
            print self.configuration
            for line in config_contents:
                print line








