import DP as dp

import numpy as np
from corels import RuleList, CorelsClassifier
from utils_greedy import *

"""
Subclass of the CORELSClassifier class, training a rule list using a greedy method with DP with exponential mechanism.
"""
class DPGreedyRLClassifier(CorelsClassifier):

    def __init__(self, max_card=2, min_support=0.01, max_length=1000000, allow_negations=True, epsilon=1, delta = None, verbosity=[], seed = -1):
        self.max_card = max_card
        self.min_support = min_support
        self.max_length = max_length
        self.allow_negations = allow_negations
        self.verbosity = verbosity
        self.status = 3
        
        DP.set_seed(seed)
        
        #For (epsilon, delta)-DP
        self.epsilon = epsilon #total budget for DP : to be divided for the different processes
        self.delta = delta
        self.gamma = 2
        self.beta = self.epsilon/ (2*self.max_length)
        
    def fit(self, X, y, features=[], prediction_name="prediction", time_limit=None, memory_limit=None, perform_post_pruning=False):
        if not (memory_limit is None):
            import os, psutil

        if not (time_limit is None):
            import time
            start = time.process_time() #clock()

        self.status = 0
        max_card = self.max_card
        min_support = self.min_support
        max_length = self.max_length
        allow_negations = self.allow_negations
        verbosity = self.verbosity
       
        rules = [] # will contain the list of lists of antecedents for each rule
        preds = [] # will contain the list of predictions for each rule
        cards = [] # will contain the list of per-class training examples cardinalities for each rule

        n_samples = y.size
        if (self.delta is None) : self.delta =0  #1 / n_samples**2 	#set delta to polynomial if not set
        #print("DP aimed : ({0},{1})".format(self.epsilon, self.delta)) 
        n_features = X.shape[1]

        stop = False # early stopping if no more rule can be found that satisfies the min. support constraint before the max. depth is reached

        X_remain = np.copy(X)
        y_remain = np.copy(y)

        # Pre-mining of the rules (takes into account min support)
        list_of_rules, tot_rules = mine_rules_combinations(X, max_card, min_support, allow_negations, features, verbosity)

        while (len(rules) < max_length) and (not stop) and (self.status == 0):
        
            #print("Computing rule number ",len(rules))
            # Greedy choice for next rule
            average_outcome_remaining = np.average(y_remain)
            init_gini =  1 - (average_outcome_remaining)**2 - (1 - average_outcome_remaining)**2 # value if no rule is added
            #print("Initial gini: ", best_gini)
            best_capt_gini = (1 - (average_outcome_remaining)**2 - (1 - average_outcome_remaining)**2) # only used to compare in case of equality
            best_rule = -1
            best_pred = -1
            best_rule_capt_indices = -1
            
            
            current_rules = list_of_rules.copy()
            info_rule = np.zeros((len(current_rules),2))
            capt_indices_rules = [[] for i in range(len(current_rules))]
            utility = np.zeros(len(current_rules))
            idx = 0
            sensitivity = 0.5 #dp.smooth_sensitivity_gini(len(X_remain),self.beta, min_supp = 1) 
            for i in range(len(current_rules)): # uses a copy of the full version as is before iterating as the list is then modified during iterations
                # Check memory limit
                a_rule = current_rules[i]
                if not (memory_limit is None):
                    mem_used = (psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)
                    if mem_used > memory_limit:
                        self.status = 5
                        break
                #Check time limit
                if not (time_limit is None):
                    end = time.process_time() #clock()
                    if end - start > time_limit:
                        self.status = 4
                        break

                rule_capt_indices = rule_indices(a_rule, X_remain) #np.where(X_remain[:,a_rule] == 1)
                n_samples_rule = rule_capt_indices[0].size  #number of samples captured by the rule
                n_samples_remain = y_remain.size
                n_samples_other = n_samples_remain - n_samples_rule #number of samples not captured yet
                # Minimum support check
                if (n_samples_rule/n_samples) > 0:
                    average_outcome_rule = np.average(y_remain[rule_capt_indices]) #clever way to know if more samples of label 0 or 1 are captured
                    pred = 0 if average_outcome_rule < 0.5 else 1
                    if len(np.delete(y_remain, rule_capt_indices)) == 0: #to avoid computing empty mean (numpy warning)
                        other_gini =0
                    else :                         
                        average_outcome_other = np.average(np.delete(y_remain, rule_capt_indices))
                        other_gini = (n_samples_other/n_samples_remain) * (1 - (average_outcome_other)**2 - (1 - average_outcome_other)**2)
                   
                    #rule_gini = 1 - (average_outcome_rule)**2 - (1 - average_outcome_rule)**2
                    capt_gini = (n_samples_rule/n_samples_remain) * (1 - (average_outcome_rule)**2 - (1 - average_outcome_rule)**2)
                    rule_gini = capt_gini + other_gini               
                    #is_different_from_default =  (pred == 0 and average_outcome_other >= 0.5) or (pred == 1 and average_outcome_other < 0.5) # not used for now
                    if rule_gini > init_gini: #if the rule does not better the model drop it
                        list_of_rules.remove(a_rule)
                        
                    else :
                        info_rule[idx]= [i, pred] #keep track of captured indexes and rule 
                        capt_indices_rules[idx] = rule_capt_indices
                        utility[idx] = rule_gini
                        idx +=1  #only increment if rule is kept for the 'Rashomon'-like set
                else:
                    list_of_rules.remove(a_rule) # the rule does not catch anything
                

                
            info_rule = info_rule[:idx]
            utility = utility[:idx] #truncate to last element idx
            
            if idx  == 0:  #means that utility is empty
                stop = True 
                
            else:
                #print("Number of rules to sample from : ", len(utility))
                best_idx = dp.exponential(self.beta, sensitivity, 1-utility)[0]
                best_rule, best_pred  = current_rules[int(info_rule[best_idx][0])], info_rule[best_idx][1]
                best_rule_capt_indices = capt_indices_rules[best_idx][0]

                
                rules.append(best_rule)
                preds.append(best_pred)

                capt_labels_counts = np.unique(y_remain[best_rule_capt_indices], return_counts=True)
                if capt_labels_counts[0].size == 2:
                    cards.append(capt_labels_counts[1])
                else:
                    if capt_labels_counts[0][0] == 0:
                        cards.append([capt_labels_counts[1][0], 0])
                    else:
                        cards.append([0, capt_labels_counts[1][0]])
                X_remain = np.delete(X_remain, best_rule_capt_indices, axis=0)
                y_remain = np.delete(y_remain, best_rule_capt_indices)
                list_of_rules.remove(best_rule)
                if(len(X_remain) < min_support) : stop = True
            
        # default rule
        if y_remain.size > 0:
            capt_labels_counts = np.unique(y_remain, return_counts=True)
            if capt_labels_counts[0].size == 2:
                cards.append(capt_labels_counts[1])
            else:
                if capt_labels_counts[0][0] == 0:
                    cards.append([capt_labels_counts[1][0], 0])
                else:
                    cards.append([0, capt_labels_counts[1][0]])
            average_outcome_rule = np.mean(y_remain)
            if average_outcome_rule < 0.5:
                pred = 0
            else:
                pred = 1
            rules.append([0])
            preds.append(pred)
        else: # No training data at all fall into the default prediction, then by default predict overall majority
            average_outcome_rule = np.average(y)
            if average_outcome_rule < 0.5:
                pred = 0
            else:
                pred = 1
            cards.append([0,0])
            rules.append([0])
            preds.append(pred)

        # Post-processing step: remove useless rules that do no change the classification function (i.e. rules before the default decision with the same prediction)
        if perform_post_pruning:
            initial_length = len(rules)
            for nomatter in range(initial_length): # just to be sure to perform enough steps
                if len(rules) > 1:
                    if preds[len(rules) - 2] == preds[len(rules) - 1]:
                        # need to remove the last rule (before the default one)
                        cards[len(rules) - 1][0] += cards[len(rules) - 2][0]
                        cards[len(rules) - 1][1] += cards[len(rules) - 2][1]
                        cards.pop(len(rules) - 2)
                        preds.pop(len(rules) - 2)
                        rules.pop(len(rules) - 2)
                    else:
                        break 
                
        # Builds a RuleList Python object (from pycorels)
        list_of_chosen_rules = []
        for i in range(len(rules)):
            local_rule = {}
            local_rule["antecedents"] = rules[i]
            local_rule["train_labels"] = cards[i]
            local_rule["prediction"] = bool(preds[i])
            list_of_chosen_rules.append(local_rule)

        self.rl_ = RuleList(rules=list_of_chosen_rules, features=features, prediction_name=prediction_name)
        if self.status == 0: # no memory or time limits reached during fitting
            self.status = -2

    def __str__(self):
        s = "DPGreedyRLClassifier (" + str(self.get_params()) + ")"

        if hasattr(self, "rl_"):
            s += "\n" + self.rl_.__str__()

        return s

    def get_status(self): 
        status = self.status 
        if status == 0:
            return "exploration running"
        elif status == 3:
            return "not_fitted"
        elif status == -2:
            return "fitted"
        elif status == 4:
            return "time_out"
        elif status == 5:
            return "memory_out"
        else:
            return "unknown"

    def get_params(self):
        """
        Get a list of all the model's parameters.
        
        Returns
        -------
        params : dict
            Dictionary of all parameters, with the names of the parameters as the keys
        """

        return {
            "max_card": self.max_card,
            "min_support": self.min_support,
            "max_length": self.max_length,
            "allow_negations": self.allow_negations
        }
