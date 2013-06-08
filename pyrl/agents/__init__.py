
# Author: Pierre-Luc Bacon <pierrelucbacon@gmail.com>

__all__ = ["skeleton_agent", "stepsizes"]

try:
    import sklearn
    __all__.append("modelbased")
except:
    pass

try:
    import pyrl.basis.tilecode
    __all__.append("qlearning")
    __all__.append("delayed_qlearning")
    __all__.append("sarsa_lambda")
    __all__.append("lstd")
    __all__.append("policy_gradient")
    __all__.append("mirror_descent")
except:
    pass

try:
    import neurolab
    __all__.append("sarsa_lambda_ann")
except:
    pass

