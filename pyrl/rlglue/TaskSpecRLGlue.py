
# There didn't appear to be any python class in place in RLGlue to allow you to 
# easily create the Task Spec string like you can in java and c++. So this is 
# my substitute so that we can be as cool as those languages. 

# VERSION <version-name> PROBLEMTYPE <problem-type> DISCOUNTFACTOR <discount-factor> 
# OBSERVATIONS INTS ([times-to-repeat-this-tuple=1] <min-value> <max-value>)* DOUBLES 
# ([times-to-repeat-this-tuple=1] <min-value> <max-value>)* CHARCOUNT <char-count> ACTIONS INTS 
# ([times-to-repeat-this-tuple=1] <min-value> <max-value>)* DOUBLES ([times-to-repeat-this-tuple=1] 
# <min-value> <max-value>)* CHARCOUNT <char-count> REWARDS (<min-value> <max-value>) EXTRA 
# [extra text of your choice goes here]";

class TaskSpec:
        def __init__(self, discount_factor=1.0, reward_range=(-1,1)):
            self.version = "RL-Glue-3.0"
            self.actions = {}
            self.observations = {}
            self.prob_type = "episodic"
            self.disc_factor = discount_factor
            self.extras = ""
            self.act_charcount = 0
            self.obs_charcount = 0
            self.reward_range = reward_range

        def toTaskSpec(self):
            ts_list = ["VERSION " + self.version, 
                       "PROBLEMTYPE " + self.prob_type, 
                       "DISCOUNTFACTOR " + str(self.disc_factor)]
            
            # Observations
            if len(self.observations.keys()) > 0:
                ts_list += ["OBSERVATIONS"]
                if self.observations.has_key("INTS"):
                    ts_list += ["INTS"] + self.observations["INTS"] 
                if self.observations.has_key("DOUBLES"):
                    ts_list += ["DOUBLES"] + self.observations["DOUBLES"] 
                if self.observations.has_key("CHARCOUNT"):
                    ts_list += ["CHARCOUNT"] + self.observations["CHARCOUNT"] 

            # Actions
            if len(self.actions.keys()) > 0:
                ts_list += ["ACTIONS"]
                if self.actions.has_key("INTS"):
                    ts_list += ["INTS"] + self.actions["INTS"] 
                if self.actions.has_key("DOUBLES"):
                    ts_list += ["DOUBLES"] + self.actions["DOUBLES"] 
                if self.actions.has_key("CHARCOUNT"):
                    ts_list += ["CHARCOUNT"] + self.actions["CHARCOUNT"] 

            ts_list += ["REWARDS", "(" + str(self.reward_range[0]) + " " + str(self.reward_range[1]) + ")"]
            if self.extras != "":
                ts_list += ["EXTRAS", self.extras]
            return ' '.join(ts_list)
            
        
        def addAction(self, dRange, repeat=1, type="INTS"):
            rept = "" if repeat<= 1 else str(repeat) + " "
            self.actions.setdefault(type, []).append("(" + rept + str(dRange[0]) + " " + str(dRange[1]) + ")")

        def addContinuousAction(self, dRange, repeat=1):
            self.addAction(dRange, repeat, "DOUBLES")

        def addDiscreteAction(self, dRange, repeat=1):
            self.addAction(map(int, dRange), repeat, "INTS")

        def addObservation(self, dRange, repeat=1, type="INTS"):
            rept = "" if repeat<= 1 else str(repeat) + " "
            self.observations.setdefault(type, []).append("(" + rept + str(dRange[0]) + " " + str(dRange[1]) + ")")

        def addContinuousObservation(self, dRange, repeat=1):
            self.addObservation(dRange, repeat, "DOUBLES")

        def addDiscreteObservation(self, dRange, repeat=1):
            self.addObservation(map(int, dRange), repeat, "INTS")

        def setActionCharLimit(self, charLimit):
            self.actions["CHARCOUNT"] = [str(charLimit)]

        def setObservationCharLimit(self, charLimit):
            self.observations["CHARCOUNT"] = [str(charLimit)]

        def setContinuing(self):
            self.prob_type = "continuing"

        def setEpisodic(self):
            self.prob_type = "episodic"

        def setDiscountFactor(self, factor):
            self.disc_factor = factor

        def setExtra(self, strExtra):
            self.extras = strExtra

        def setProblemTypeCustom(self, strProbType):
            self.prob_type = strProbType

        def setRewardRange(self, low, high):
            self.reward_range = (low, high)

