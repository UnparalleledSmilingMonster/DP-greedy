from corels import load_from_csv, RuleList, CorelsClassifier
from HeuristicRL import GreedyRLClassifier
from HeuristicRL_DP import DPGreedyRLClassifier
from HeuristicRL_DP_smooth import DpSmoothGreedyRLClassifier
import numpy as np
import DP as dp


dataset = "compas"
min_support = 0.20
max_length = 5
max_card = 3
epsilon = 2
compute_exact = False
verbosity = [] # ["mine"] # ["mine"]
X, y, features, prediction = load_from_csv("data/%s.csv" %dataset)

X_unbias,features_unbias = dp.clean_dataset(X,features,dp.get_biases(dataset))
print(list(set(features)-set(features_unbias)))


N_runs = 10
res = np.zeros(N_runs)
for i in range(N_runs):
    if not compute_exact:
        # Greedy
        greedy_rl = DpSmoothGreedyRLClassifier(min_support=min_support, max_length=max_length, verbosity=verbosity, max_card=max_card, allow_negations=True, epsilon = epsilon, noise = "Cauchy")
        greedy_rl.fit(X_unbias, y, features=features_unbias, prediction_name=prediction)
        my_rl = greedy_rl
    else:
        # CORELS
        corels_rl = CorelsClassifier(min_support=0.00, max_length=max_length, verbosity=verbosity, max_card=max_card, c=0.0, n_iter=1000000)
        corels_rl.fit(X, y, features=features, prediction_name=prediction)
        my_rl = corels_rl
    
    res[i]= np.average(my_rl.predict(X_unbias) == y)
    
print(my_rl)


greedy_rl = GreedyRLClassifier(min_support=min_support, max_length=max_length, verbosity=verbosity, max_card=max_card, allow_negations=True)
greedy_rl.fit(X_unbias, y, features=features_unbias, prediction_name=prediction)
my_rl = greedy_rl



f = open("DP_results/cauchy_smooth.txt", "a")
f.write("\nCauchy Smooth DP : epsilon={0}, max_length={1}, max_card={2}, min_supp={3}\n".format(epsilon, max_length, max_card, min_support))
f.write("Number of runs: {0}\n".format(N_runs))
f.write("Average accuracy: {0}\n".format(np.mean(res)))
f.write("Variance of accuracy: {0}\n".format(np.var(res)))
f.write("Min accuracy: {0}\n".format(np.min(res)))
f.write("Max accuracy: {0}\n".format(np.max(res)))
f.write("####################\n")
f.write("Vanilla Greedy RL : acc={0}\n".format(np.average(my_rl.predict(X_unbias) == y)))
f.close()




"""
train_acc = np.average(my_rl.predict(X) == y)
print("train_acc = ", train_acc)
print("Search status = ", my_rl.get_status())
"""



#for i in range(len(rules)):
#    if i > 0:
#        print("else ", end = '')
#    if rules[i] >= 0:
#        print("if " + features[rules[i]] + " then " + prediction + "=" + str(preds[i]))
#    else:
#        print(prediction + "=" + str(preds[i]))





