from corels import load_from_csv, RuleList, CorelsClassifier
from HeuristicRL import GreedyRLClassifier
from HeuristicRL_DP import DPGreedyRLClassifier
from HeuristicRL_DP_smooth import DpSmoothGreedyRLClassifier
import numpy as np
import DP as dp

from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

from art.attacks.inference.membership_inference import MembershipInferenceBlackBox, MembershipInferenceBlackBoxRuleBased
from art.estimators.estimator import BaseEstimator
from art.estimators.classification import BlackBoxClassifier
from art.metrics.privacy.worst_case_mia_score import get_roc_for_fpr

              
dataset = "german_credit"
min_support = 0.15
max_length = 10
max_card = 1
epsilon = 1
verbosity = [] # ["mine"] # ["mine"]
X, y, features, prediction = load_from_csv("data/%s.csv" %dataset)
seed = 42
X_unbias,features_unbias = dp.clean_dataset(X,features, dataset)
N = len(X_unbias)            
x_train, y_train, x_test, y_test= dp.split_dataset(X_unbias, y, 0.50, seed =seed)

def wrap_predict(model, X):
    predictions =  model.predict(X)
    
    #One hot encoding
    res = np.zeros((len(predictions),2))
    for i in range(len(predictions)):
        if predictions[i] == True:
            res[i]=[0,1]
        else:
            res[i]=[1,0]
    
    return res



def MIA_rule_list(model, x_train, y_train, x_test, y_test, attack_train_ratio = 0.7):
    # train attack model
    attack_train_size = int(len(x_train) * attack_train_ratio)
    attack_test_size = int(len(x_test) * attack_train_ratio)

    wrapper = BlackBoxClassifier(lambda X : wrap_predict(model, X), input_shape = x_train[0].shape, nb_classes = 2)
    bb_attack = MembershipInferenceBlackBox(wrapper, attack_model_type = "rf") #we use a random forest as the attack model

    bb_attack.fit(x_train[:attack_train_size].astype(np.float32), y_train[:attack_train_size], x_test[:attack_test_size].astype(np.float32), y_test[:attack_test_size])
                  
    # get inferred values
    inferred_train_bb = bb_attack.infer(x_train[attack_train_size:].astype(np.float32), y_train[attack_train_size:])
    inferred_test_bb = bb_attack.infer(x_test[attack_test_size:].astype(np.float32), y_test[attack_test_size:])
    # check accuracy
    train_acc = np.sum(inferred_train_bb) / len(inferred_train_bb)
    
    def infer_reverse(model, train_acc, x, y):
        """ to use if model is under the roc curve"""
        if train_acc<0.5:
            return 1 - bb_attack.infer(x, y, probabilities=True)
        return bb_attack.infer(x, y, probabilities=True)

        
    test_acc = 1 - (np.sum(inferred_test_bb) / len(inferred_test_bb))
    acc = (train_acc * len(inferred_train_bb) + test_acc * len(inferred_test_bb)) / (len(inferred_train_bb) + len(inferred_test_bb))
    print("\nBlack Box Attack:")
    print(f"Members Accuracy: {train_acc:.4f}")
    print(f"Non Members Accuracy {test_acc:.4f}")
    print(f"Attack Accuracy {acc:.4f}")


    # we run the worst case metric on trainset to find an appropriate threshold
    # Black Box
    bb_members_test_prob = bb_attack.infer(x_train[attack_train_size:].astype(np.float32), y_train[attack_train_size:], probabilities=True)
    bb_nonmembers_test_prob = bb_attack.infer(x_test[attack_test_size:].astype(np.float32), y_test[attack_test_size:], probabilities=True)

    bb_mia_test_probs = np.concatenate((np.squeeze(bb_members_test_prob, axis=-1),
                                   np.squeeze(bb_nonmembers_test_prob, axis=-1)))
                                  
    bb_mia_test_labels = np.concatenate((np.ones_like(y_train[attack_train_size:]), np.zeros_like(y_test[attack_test_size:])))

    # We allow 1% FPR 
    """
    fpr, tpr, threshold = get_roc_for_fpr(attack_proba=bb_mia_test_probs, attack_true=bb_mia_test_labels, targeted_fpr=0.001)[0]
    print(f'{tpr=}: {fpr=}: {threshold=}')"""     
    
    fpr, tpr, _ = roc_curve( y_true=bb_mia_test_labels, y_score=bb_mia_test_probs, drop_intermediate = False)
    plt.figure(figsize=(8,8))
    print(tpr)
    for i in range(len(fpr)):
        if tpr[i] < fpr[i] : tpr[i],fpr[i] = fpr[i], tpr[i]
    plt.plot(fpr, tpr, color="darkorange", linewidth =2, label="ROC curve")
    plt.plot([0, 1], [0, 1], color="navy", linewidth =2, linestyle="--", label='Random Inference')
    #plt.xlim([0.0, 1.0])
    #plt.ylim([0.0, 1.0])
    plt.xticks(np.linspace(0,1,10))
    plt.xlabel("False Positive Rate")
    plt.xscale("symlog")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    plt.show()
    

corels_rl = CorelsClassifier(n_iter=500000, map_type="prefix", policy="objective", verbosity=["rulelist"], ablation=0, max_card=max_card, min_support=0.01, max_length=100, c=0.0000001)
corels_rl.fit(x_train, y_train, features=features_unbias, prediction_name=prediction)
train_acc = np.average(corels_rl.predict(x_train) == y_train)
test_acc = np.average(corels_rl.predict(x_test) == y_test)
print("DP Smooth RL:")
print("train_acc= ", train_acc)
print("test_acc = ", test_acc)
print(corels_rl.get_status())

MIA_rule_list(corels_rl, x_train, y_train, x_test, y_test)

print("\n####################\n")


    


dp_smooth_rl = DpSmoothGreedyRLClassifier(min_support=min_support, max_length=max_length, verbosity=verbosity, max_card=max_card, allow_negations=True, epsilon = epsilon, noise = "Laplace", seed = seed)
dp_smooth_rl.fit(x_train, y_train, features=features_unbias, prediction_name=prediction)
train_acc = np.average(dp_smooth_rl.predict(x_train) == y_train)
test_acc = np.average(dp_smooth_rl.predict(x_test) == y_test)
print("DP Smooth RL:")
print("train_acc= ", train_acc)
print("test_acc = ", test_acc)

MIA_rule_list(dp_smooth_rl, x_train, y_train, x_test, y_test)

print("\n####################\n")

#Part for greedy rl vanilla          

greedy_rl = GreedyRLClassifier(min_support=0.0, max_length=max_length, verbosity=verbosity, max_card=max_card, allow_negations=True)
greedy_rl.fit(x_train, y_train, features=features_unbias, prediction_name=prediction)
train_acc = np.average(greedy_rl.predict(x_train) == y_train)
test_acc = np.average(greedy_rl.predict(x_test) == y_test)
print("Vanilla Greedy RL:")
print("train_acc= ", train_acc)
print("test_acc = ", test_acc)

MIA_rule_list(greedy_rl, x_train, y_train, x_test, y_test)

