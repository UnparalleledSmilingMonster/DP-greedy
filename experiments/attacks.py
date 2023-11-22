from corels import load_from_csv, RuleList, CorelsClassifier
from HeuristicRL import GreedyRLClassifier
from HeuristicRL_DP import DPGreedyRLClassifier
from HeuristicRL_DP_smooth import DpSmoothGreedyRLClassifier
import numpy as np
import DP as dp





from art.attacks.inference.membership_inference import MembershipInferenceBlackBox, MembershipInferenceBlackBoxRuleBased
from art.estimators.estimator import BaseEstimator
from art.estimators.classification.classifier import ClassifierMixin
from art.estimators.classification import BlackBoxClassifier


              
              

dataset = "adult"
min_support = 0.10
max_length = 5
max_card = 2
epsilon = 0.1
verbosity = [] # ["mine"] # ["mine"]
X, y, features, prediction = load_from_csv("data/%s.csv" %dataset)
seed = 1


X_unbias,features_unbias = dp.clean_dataset(X,features, dp.get_biases(dataset))
print(list(set(features)-set(features_unbias)))

ratio = int(len(X_unbias)*0.70)
x_train, y_train = X_unbias[:ratio].astype(float), y[:ratio]
x_test, y_test  = X_unbias[ratio:].astype(float), y[ratio:]



greedy_rl = DpSmoothGreedyRLClassifier(min_support=min_support, max_length=max_length, verbosity=verbosity, max_card=max_card, allow_negations=True, epsilon = epsilon, noise = "Laplace", seed = seed)
greedy_rl.fit(x_train, y_train, features=features_unbias, prediction_name=prediction)
my_rl = greedy_rl
train_acc = np.average(my_rl.predict(x_train) == y_train)
test_acc = np.average(my_rl.predict(x_test) == y_test)
print(my_rl)
print("train_acc= ", train_acc)
print("test_acc = ", test_acc)


def wrap_predict(model, X):
    predictions =  my_rl.predict(X)
    
    #One hot encoding
    res = np.zeros((len(predictions),2))
    for i in range(len(predictions)):
        if predictions[i] == True:
            res[i]=[0,1]
        else:
            res[i]=[1,0]
    
    return res
    

arr = np.concatenate((x_test,y_test.reshape(-1,1)), axis = 1)
wrapper = BlackBoxClassifier(lambda X : wrap_predict(my_rl, X), input_shape = x_train[0].shape, nb_classes = 2)


attack = MembershipInferenceBlackBoxRuleBased(wrapper)

# infer attacked feature
inferred_train = attack.infer(x_train, y_train)
inferred_test = attack.infer(x_test, y_test)

# check accuracy
train_acc = np.sum(inferred_train) / len(inferred_train)
test_acc = 1 - (np.sum(inferred_test) / len(inferred_test))
acc = (train_acc * len(inferred_train) + test_acc * len(inferred_test)) / (len(inferred_train) + len(inferred_test))
print(f"Members Accuracy: {train_acc:.4f}")
print(f"Non Members Accuracy {test_acc:.4f}")
print(f"Attack Accuracy {acc:.4f}")


"""

bb_attack = MembershipInferenceBlackBox(wrapper)

# train attack model
attack_train_ratio = 0.5
attack_train_size = int(len(x_train) * attack_train_ratio)
attack_test_size = int(len(x_test) * attack_train_ratio)

bb_attack.fit(x_train[:attack_train_size], y_train[:attack_train_size],
              x_test[:attack_test_size], y_test[:attack_test_size])
              
# get inferred values
inferred_train_bb = bb_attack.infer(x_train[attack_train_size:], y_train[attack_train_size:])
inferred_test_bb = bb_attack.infer(x_test[attack_test_size:], y_test[attack_test_size:])
# check accuracy
train_acc = np.sum(inferred_train_bb) / len(inferred_train_bb)
test_acc = 1 - (np.sum(inferred_test_bb) / len(inferred_test_bb))
acc = (train_acc * len(inferred_train_bb) + test_acc * len(inferred_test_bb)) / (len(inferred_train_bb) + len(inferred_test_bb))
print(f"Members Accuracy: {train_acc:.4f}")
print(f"Non Members Accuracy {test_acc:.4f}")
print(f"Attack Accuracy {acc:.4f}")

"""          


greedy_rl = GreedyRLClassifier(min_support=0.0, max_length=max_length, verbosity=verbosity, max_card=max_card, allow_negations=True)
greedy_rl.fit(X_unbias, y, features=features_unbias, prediction_name=prediction)
my_rl = greedy_rl
test_acc = np.average(my_rl.predict(x_test) == y_test)
print("test_acc = ", test_acc)


wrapper = BlackBoxClassifier(lambda X : wrap_predict(greedy_rl, X), input_shape = x_train[0].shape, nb_classes = 2)

attack = MembershipInferenceBlackBoxRuleBased(wrapper)

# infer attacked feature
inferred_train = attack.infer(x_train, y_train)
inferred_test = attack.infer(x_test, y_test)

# check accuracy
train_acc = np.sum(inferred_train) / len(inferred_train)
test_acc = 1 - (np.sum(inferred_test) / len(inferred_test))
acc = (train_acc * len(inferred_train) + test_acc * len(inferred_test)) / (len(inferred_train) + len(inferred_test))
print(f"Members Accuracy: {train_acc:.4f}")
print(f"Non Members Accuracy {test_acc:.4f}")
print(f"Attack Accuracy {acc:.4f}")


"""
bb_attack = MembershipInferenceBlackBox(wrapper)

# train attack model
attack_train_ratio = 0.5
attack_train_size = int(len(x_train) * attack_train_ratio)
attack_test_size = int(len(x_test) * attack_train_ratio)

bb_attack.fit(x_train[:attack_train_size], y_train[:attack_train_size],
              x_test[:attack_test_size], y_test[:attack_test_size])
              
# get inferred values
inferred_train_bb = bb_attack.infer(x_train[attack_train_size:], y_train[attack_train_size:])
inferred_test_bb = bb_attack.infer(x_test[attack_test_size:], y_test[attack_test_size:])
# check accuracy
train_acc = np.sum(inferred_train_bb) / len(inferred_train_bb)
test_acc = 1 - (np.sum(inferred_test_bb) / len(inferred_test_bb))
acc = (train_acc * len(inferred_train_bb) + test_acc * len(inferred_test_bb)) / (len(inferred_train_bb) + len(inferred_test_bb))
print(f"Members Accuracy: {train_acc:.4f}")
print(f"Non Members Accuracy {test_acc:.4f}")
print(f"Attack Accuracy {acc:.4f}")
"""

