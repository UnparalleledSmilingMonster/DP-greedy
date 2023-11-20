from corels import load_from_csv, RuleList, CorelsClassifier
from HeuristicRL import GreedyRLClassifier
from HeuristicRL_DP import DPGreedyRLClassifier
from HeuristicRL_DP_smooth import DpSmoothGreedyRLClassifier
import numpy as np
import DP as dp



from art.attacks.inference.membership_inference.black_box_rule_based import MembershipInferenceBlackBoxRuleBased
from art.estimators.estimator import BaseEstimator
from art.estimators.classification.classifier import ClassifierMixin



dataset = "compas"
min_support = 0.20
max_length = 5
max_card = 2
epsilon = 1
verbosity = [] # ["mine"] # ["mine"]
X, y, features, prediction = load_from_csv("data/%s.csv" %dataset)
seed = 42


X_unbias,features_unbias = dp.clean_dataset(X,features, dp.get_biases(dataset))
print(list(set(features)-set(features_unbias)))


greedy_rl = DpSmoothGreedyRLClassifier(min_support=min_support, max_length=max_length, verbosity=verbosity, max_card=max_card, allow_negations=True, epsilon = epsilon, noise = "Laplace", seed = seed)
greedy_rl.fit(X_unbias, y, features=features_unbias, prediction_name=prediction)
my_rl = greedy_rl
train_acc = np.average(my_rl.predict(X_unbias) == y)
print("train_acc = ", train_acc)


"""
greedy_rl = GreedyRLClassifier(min_support=0.0, max_length=max_length, verbosity=verbosity, max_card=max_card, allow_negations=True)
greedy_rl.fit(X_unbias, y, features=features_unbias, prediction_name=prediction)
my_rl = greedy_rl
train_acc = np.average(my_rl.predict(X_unbias) == y)
print("train_acc = ", train_acc)
"""



