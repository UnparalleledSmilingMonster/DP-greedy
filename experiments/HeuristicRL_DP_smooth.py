import DP as dp

import numpy as np
from corels import RuleList, CorelsClassifier
from utils_greedy import *

"""
Subclass of the CORELSClassifier class, training a rule list using a greedy method with DP with noisy max report (cauchy noise)."""

class DpSmoothGreedyRLClassifier(CorelsClassifier):

    def __init__(self, max_card=2, min_support=0.01, max_length=1000000, allow_negations=True, epsilon=1, delta = None, confidence = 0.98, noise = "Cauchy", verbosity=[], seed = 42):
        self.max_card = max_card
        self.min_support = min_support
        self.max_length = max_length
        self.allow_negations = allow_negations
        self.verbosity = verbosity
        self.status = 3
        #For (epsilon, delta)-DP
        self.epsilon = epsilon #total budget for DP : to be divided for the different processes
        self.delta = delta
        self.noise = noise
        self.budget_per_node = self.epsilon / (3*self.max_length-1)
        self.confidence = confidence
        
        self.threshold = dp.confidence_interval_laplace(self.budget_per_node, self.confidence)
        
        dp.set_seed(seed)
        
        self.true_cards=[] #used for distributional overfitting computation : NOT DP (not meant for release, at all)!
        
        if self.noise == "Cauchy":
            self.gamma = 2
            self.beta = self.budget_per_node/(2*(self.gamma+1))
            self.delta =0  #pure DP with Cauchy Noise
            
        

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
        n_features = X.shape[1]
        min_supp_N = np.floor(self.min_support * n_samples)
        
        if self.noise == "Laplace":
            if (self.delta is None or self.delta == "None") : self.delta =1 / (n_samples**2 * (self.max_length-1))	#set delta to polynomial if not set
            self.beta = self.budget_per_node/(2*np.log(2/self.delta))
            
            
        #print("DP aimed : ({0},{1})".format(self.epsilon, self.delta)) 
        
        
        
        stop = False # early stopping if no more rule can be found that satisfies the min. support constraint before the max. depth is reached

        X_remain = np.copy(X)
        y_remain = np.copy(y)

        # Pre-mining of the rules (takes into account min support)
        list_of_rules, tot_rules = mine_rules_combinations(X, max_card, min_support, allow_negations, features, verbosity)

        while (len(rules) < max_length -1) and (not stop) and (self.status == 0):
            # Greedy choice for next rule    
            
            stop=True
            if len(X_remain) + dp.laplace(self.budget_per_node,1,1)[0] >= min_supp_N + self.threshold: #1st dp mechanism
                
                stop = False
                confident_X_remain = max(len(X_remain),min_supp_N)
                smooth_sensitivity = dp.smooth_sensitivity_gini(confident_X_remain,self.beta, min_supp = min_supp_N)     
                average_outcome_remaining = np.average(y_remain)
                best_gini =  1 - (average_outcome_remaining)**2 - (1 - average_outcome_remaining)**2 # value if no rule is added$

                if self.noise=="Cauchy": best_gini +=dp.cauchy_smooth(self.beta, self.gamma, smooth_sensitivity) #noisy version
                else : best_gini += dp.laplace_smooth(self.budget_per_node, smooth_sensitivity) #noisy version
                
                best_capt_gini = (1 - (average_outcome_remaining)**2 - (1 - average_outcome_remaining)**2) # only used to compare in case of equality
                best_rule = -1
                best_rule_capt_indices = -1
            
                current_rules = list_of_rules.copy()
                
                for i in range(len(current_rules)): # uses a copy of the full version as is before iterating as the list is then modified during iterations      
                    a_rule = current_rules[i]
                    # Check memory limit
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


                    if len(y_remain) == len(rule_capt_indices[0]): #to avoid computing empty mean (numpy warning)
                        other_gini =0
                    else :                         
                        average_outcome_other = np.average(np.delete(y_remain, rule_capt_indices))
                        other_gini = (n_samples_other/n_samples_remain) * (1 - (average_outcome_other)**2 - (1 - average_outcome_other)**2)
                   
                    if len(rule_capt_indices[0]) ==0 :
                        capt_gini = 0     
                    else :                   
                        average_outcome_rule = np.average(y_remain[rule_capt_indices]) #to know if more samples of label 0 or 1 are captured
                        capt_gini = (n_samples_rule/n_samples_remain) * (1 - (average_outcome_rule)**2 - (1 - average_outcome_rule)**2)
                    rule_gini = capt_gini + other_gini
                    
                    
                    if self.noise == "Cauchy":
                        rule_gini += dp.cauchy_smooth(self.beta, self.gamma, smooth_sensitivity) #noisy version
                    else : 
                        rule_gini += dp.laplace_smooth(self.budget_per_node, smooth_sensitivity) #noisy version
                    #is_different_from_default =  (pred == 0 and average_outcome_other >= 0.5) or (pred == 1 and average_outcome_other < 0.5) # not used for now
                    if (rule_gini < best_gini) or \
                        ((rule_gini == best_gini) and (capt_gini < best_capt_gini)):
                        #print("-> new gini: ", rule_gini)
                        #best_different_from_default = is_different_from_default # not used for now
                        best_gini = rule_gini
                        best_capt_gini = capt_gini # used to select the best "side of the split" (most accurate rule if two splits allows the same children-summed gini impurity reduction)
                        best_rule = a_rule
                        best_rule_capt_indices = rule_capt_indices
                        
                if best_rule == -1: # no rule OK found
                    stop = True 
                else:
                    count0_noisy, count1_noisy = self.get_noisy_counts(y_remain, best_rule_capt_indices)
                    best_pred = DpSmoothGreedyRLClassifier.best_pred(count0_noisy, count1_noisy)
                    cards.append([count0_noisy, count1_noisy])                   
                    rules.append(best_rule)
                    preds.append(best_pred)              
                    X_remain = np.delete(X_remain, best_rule_capt_indices, axis=0)
                    y_remain = np.delete(y_remain, best_rule_capt_indices)
                    list_of_rules.remove(best_rule)
                  
                
            
        # default rule
        count0_noisy, count1_noisy = self.get_noisy_counts(y_remain, None)
        best_pred = DpSmoothGreedyRLClassifier.best_pred(count0_noisy, count1_noisy)
        cards.append([count0_noisy, count1_noisy])                   
        rules.append([0])
        preds.append(best_pred)              

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

    def get_noisy_counts(self, y_remain, rule_capt_indices):    
        if rule_capt_indices is None :
            capt_labels_counts = np.unique(y_remain, return_counts=True)
        capt_labels_counts = np.unique(y_remain[rule_capt_indices], return_counts=True)
        
        if capt_labels_counts[0].size == 2:                                    
            capt_labels_0 = capt_labels_counts[1][0]
            capt_labels_1 = capt_labels_counts[1][1]        
        elif len(capt_labels_counts[0]) == 0 :
            capt_labels_0 = 0
            capt_labels_1 = 0
            
        else:
            if capt_labels_counts[0][0] == 0:
                capt_labels_0 = capt_labels_counts[1][0]
                capt_labels_1 = 0
                
            else:
                capt_labels_0 = 0
                capt_labels_1 = capt_labels_counts[1][0]
        
        self.true_cards.append([capt_labels_0, capt_labels_1])
        
        capt_labels_0 += dp.laplace(self.budget_per_node, 1, 1)[0]
        capt_labels_1 += dp.laplace(self.budget_per_node, 1, 1)[0]
        
        return capt_labels_0, capt_labels_1
        
    @staticmethod
    def best_pred(count0, count1):
        return 0 if count0 >= count1 else 1 
        
    def __str__(self):
        s = "DP with Smooth Sensitivity GreedyRLClassifier (" + str(self.get_params()) + ")"
       

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
            "allow_negations": self.allow_negations,
            "epsilon": self.epsilon,
            "delta": self.delta
        }
