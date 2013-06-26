
import collections, random
import argparse
import numpy


class ValueRange(collections.Container):
    '''A class wrapping a list with some extra functional magic, like head,
    tail, init, last, drop, and take.'''

    def __init__(self, min=0., max=1., dtype=float):
        self.dtype = dtype
        if dtype is not int and dtype is not float:
            raise TypeError("data type not understood")
        self.min_value = dtype(min)
        self.max_value = dtype(max)

    def __len__(self):
        return self.max_value - self.min_value

    def __getitem__(self, key):
        # Takes a key value between 0 and 1 and converts that into a
        # value in the range
        #key = numpy.clip(key, self.min_value, self.max_value)
        #return self.__rescale(key, self.min_value, self.max_value)
        raise IndexError()

    def __contains__(self, value):
        return (value >= self.min_value) & (value <= self.max_value)

    def __rescale(self, value, min_val, max_val):
        return value * (max_val - min_val) + min_val

    def min(self):
        return self.min_value

    def max(self):
        return self.max_value

    def sample_rand(self, size=None):
        if self.dtype is int:
            return numpy.random.randint(self.min_value, self.max_value+1, size=size)
        else:
            return self.__rescale(numpy.random.random(size=size), self.min_value, self.max_value)

    def sample_logrand(self, size=None, tau=1.0):
        values = numpy.exp(self.__rescale(numpy.random.random(size=size), -34., 0.)/tau)
        values = self.__rescale(values, self.min_value, self.max_value)
        if size is None:
            values = self.dtype(values)
        else:
            values = numpy.array(values, dtype=self.dtype)
        return values

    def sample_exprand(self, size=None):
        values = numpy.log(self.__rescale(numpy.random.random(size=size), 1., numpy.exp(1.)))
        values = self.__rescale(values, self.min_value, self.max_value)
        if size is None:
            values = self.dtype(values)
        else:
            values = numpy.array(values, dtype=self.dtype)
        return values


def parameter_set(alg_name, **kwargs):
    kwargs['prog'] = alg_name
    kwargs['conflict_handler'] = 'resolve'
    kwargs['add_help'] = False
    parser = argparse.ArgumentParser(**kwargs)
    parser.add_argument_group(title="optimizable",
        description="Algorithm parameters that should/can be optimized. " + \
        "Only these parameters are modified during parameter optimizations.")
    return parser

def add_parameter(parser, name, min=0., max=1.0, optimize=True, **kwargs):
    # All keyword arguments besides min/max should be
    # valid keyword args for add_argument.
    # If choices specified, try to guess type from its contents
    if kwargs.has_key('nargs') and kwargs['nargs'].__class__ is not int:
        raise TypeError("Parameters only allowed to have integer number of arguments")

    if kwargs.has_key('choices'):
        kwargs.setdefault('type', kwargs['choices'][0].__class__)
    else:
        # Otherwise, default to float
        kwargs.setdefault('type', float)
        # No choices specified, so generate them based on type
        if kwargs['type'] in [int, float]:
            value_range = ValueRange(min, max, dtype=kwargs['type'])
            kwargs['choices'] = value_range
            kwargs['metavar'] = str(min) + ".." + str(max)
        elif kwargs['type'] is not bool:
            raise TypeError("String typed parameter requires 'choices' argument")

    if optimize:
        i = map(lambda k: k.title, parser._action_groups).index("optimizable")
        parser._action_groups[i].add_argument("--"+name, **kwargs)
    else:
        parser.add_argument("--"+name, **kwargs)


def remove_parameter(parser, name):
    # First find the argument/parameter itself
    argument = parser._actions[map(lambda k: k.dest, parser._actions).index(name)]
    # Next, remove it from all the _actions lists
    parser._remove_action(argument)
    # Finally, needs to be removed from the _group_actions list for its group
    for grp in parser._action_groups:
        try:
            grp._group_actions.remove(argument)
        except:
            pass

def get_optimize_group(parser):
    i = map(lambda k: k.title, parser._action_groups).index("optimizable")
    return parser._action_groups[i]

def sample_parameter(param):
    if param.type is bool:
        return bool(random.getrandbits(1))
    elif param.choices: # All other cases should be handled by sampling choices
        try:
            if param.nargs > 1:
                return param.choices.sample_rand(size=param.nargs)
            else:
                return param.choices.sample_rand()
        except:
            return numpy.random.choice(param.choices)
    else:
        # This *SHOULDNT* happen, it means it is not a boolean
        # but also has empty choices...
        raise TypeError("non-boolean parameter must have 'choices' of some container")

def randomize_parameters(parser):
    opt_grp = get_optimize_group(parser)
    param_samples = []
    opt_pnames = set()
    for param in opt_grp._group_actions:
        param_samples.append((param.dest, sample_parameter(param)))
        opt_pnames.add(param.dest)

    for param in parser._actions:
        if param.dest not in opt_pnames:
            opt_pnames.add(param.dest)
            param_samples.append((param.dest, param.default))
    return param_samples







