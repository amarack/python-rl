
from pyrl.rlglue.run import *
import argparse

# List out the step-size algorithms to run study over
stepsize_algorithms = [stepsizes.GHS, stepsizes.McClains, stepsizes.STC, 
                       stepsizes.RProp, stepsizes.Autostep, stepsizes.AlphaBounds]

# Form the agents by attatching step-size algorithm to sarsa
for ssalg in stepsize_algorithms:
    stepsizes.genAdaptiveAgent(ssalg, sarsa_lambda.sarsa_lambda)

# Create parser for command line arguments
parser = argparse.ArgumentParser(description='Run step-size study experiment.')
addRunExpArgs(parser)
args = parser.parse_args()

# Run study!
if args.load is None:
    config = fromuser()
    if args.genjson:
        print tojson(*config)
    else:
        run(*config,local=args.local, result_file=args.output)
else:
    config = fromjson(args.load)
    run(*config, local=args.local, result_file=args.output)

