from corels import load_from_csv, RuleList, CorelsClassifier
from HeuristicRL import GreedyRLClassifier
from HeuristicRL_DP import DPGreedyRLClassifier
from HeuristicRL_DP_smooth import DpSmoothGreedyRLClassifier
import numpy as np
import DP as dp

dataset = "folktable"
min_support = 0.05
max_length = 10
max_card = 2
epsilon = 1
verbosity = [] # ["mine"] # ["mine"]
X, y, features, prediction = load_from_csv("data/%s.csv" %dataset)
seed = 30
X_unbias,features_unbias = dp.clean_dataset(X,features, dataset)
N = len(X_unbias)            
x_train, y_train, x_test, y_test= dp.split_dataset(X_unbias, y, 0.80, seed =seed)

DP_smooth_rl = DpSmoothGreedyRLClassifier(min_support=min_support, max_length=max_length, verbosity=verbosity, max_card=max_card, allow_negations=True, epsilon = epsilon, noise = "Laplace", confidence=0.98)
DP_smooth_rl.fit(x_train, y_train, features=features_unbias, prediction_name=prediction)
DP_smooth_rl.distributional_overfit(x_train, x_test)

greedy_rl = GreedyRLClassifier(min_support=0.0, max_length=max_length, verbosity=verbosity, max_card=max_card, allow_negations=True)
greedy_rl.fit(x_train, y_train, features=features_unbias, prediction_name=prediction)
greedy_rl.distributional_overfit(x_train, x_test)


