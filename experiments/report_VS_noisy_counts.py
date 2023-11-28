from HeuristicRL_DP_noise import DpNoiseGreedyRLClassifier
from DP_global_old import DpNoiseGreedyRLClassifier as DpOld
from corels import load_from_csv, RuleList, CorelsClassifier
from HeuristicRL import GreedyRLClassifier
import DP as dp

import numpy as np
import matplotlib.pyplot as plt

dataset="compas"
max_card = 2


X, y, features, prediction = load_from_csv("data/%s.csv" %dataset)
X_unbias,features_unbias = dp.clean_dataset(X,features, dp.get_biases(dataset))
N = len(X_unbias)            

epsilons = np.logspace(-1,4,100)
res = np.zeros((2,len(epsilons)))
N_seeds = 20
seeds = np.arange(1,N_seeds+1,1)

for i in range(len(epsilons)) :
    epsilon = epsilons[i]
    for seed in seeds : 
        greedy_rl = DpNoiseGreedyRLClassifier(min_support=0.0, max_length=5, max_card=max_card, allow_negations=True, epsilon = epsilon, delta =None, noise = "Laplace", seed = seed) 
        greedy_rl.fit(X_unbias, y, features=features_unbias, prediction_name=prediction)  
        res[0,i] += np.average(greedy_rl.predict(X_unbias) == y)
        
        greedy_rl = DpOld(min_support=0.0, max_length=5, max_card=max_card, allow_negations=True, epsilon = epsilon, delta =None, noise = "Laplace", seed = seed) 
        greedy_rl.fit(X_unbias, y, features=features_unbias, prediction_name=prediction)  
        res[1,i] += np.average(greedy_rl.predict(X_unbias) == y)    

res /= N_seeds


greedy_rl = GreedyRLClassifier(min_support=0.0, max_length=5, max_card=max_card, allow_negations=True) 
greedy_rl.fit(X_unbias, y, features=features_unbias, prediction_name=prediction)  



plt.figure(figsize=(12,8))        
plt.plot(epsilons, res[0], label="Noisy counts")
plt.plot(epsilons, res[1], label="Noisy Gini")
plt.axhline(np.average(greedy_rl.predict(X_unbias) == y), label="baseline algo", color ="red", linestyle=":")
plt.xscale('log')
plt.xlabel(r"Privacy budget $\epsilon$", fontsize = 18)
plt.ylabel("Training accuracy", fontsize=18)
plt.title("Comparison of DP methods on compas", fontsize = 20)
plt.legend(fontsize=14)
plt.show()
