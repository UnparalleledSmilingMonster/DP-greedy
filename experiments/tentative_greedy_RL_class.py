
#class MyGreedyRL:
#    def fit(self, X, y):
from corels import load_from_csv, RuleList, CorelsClassifier
from HeuristicRL import GreedyRLClassifier
import numpy as np

dataset = "compas"
min_support = 0.05
max_length = 5
max_card = 2
compute_exact = False
verbosity = [] # ["mine"] # ["mine"]
X, y, features, prediction = load_from_csv("data/%s.csv" %dataset)

print(X.shape)

if not compute_exact:
    # Greedy
    greedy_rl = GreedyRLClassifier(min_support=min_support, max_length=max_length, verbosity=verbosity, max_card=max_card, allow_negations=True)
    greedy_rl.fit(X, y, features=features, prediction_name=prediction)
    my_rl = greedy_rl
else:
    # CORELS
    corels_rl = CorelsClassifier(min_support=min_support, max_length=max_length, verbosity=verbosity, max_card=max_card, c=0.0, n_iter=1000000)
    corels_rl.fit(X, y, features=features, prediction_name=prediction)
    my_rl = corels_rl

print(my_rl)
train_acc = np.average(my_rl.predict(X) == y)
print("train_acc = ", train_acc)
print("Search status = ", my_rl.get_status())
#for i in range(len(rules)):
#    if i > 0:
#        print("else ", end = '')
#    if rules[i] >= 0:
#        print("if " + features[rules[i]] + " then " + prediction + "=" + str(preds[i]))
#    else:
#        print(prediction + "=" + str(preds[i]))





