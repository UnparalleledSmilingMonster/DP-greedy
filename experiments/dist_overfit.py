from corels import load_from_csv, RuleList, CorelsClassifier
from HeuristicRL import GreedyRLClassifier
from HeuristicRL_DP import DPGreedyRLClassifier
from HeuristicRL_DP_smooth import DpSmoothGreedyRLClassifier
import numpy as np
import DP as dp

dataset = "german_credit"
min_support = 0.12
max_length = 7
max_card = 1
epsilon = 10
runs = 100
verbosity = [] # ["mine"] # ["mine"]

#seed = 42
#X,y = X[:1000], y[:1000]


"""
corels_rl = CorelsClassifier(n_iter=300000, map_type="prefix", policy="objective", verbosity=["rulelist"], ablation=0, max_card=max_card, min_support=0.00, max_length=10000, c=0.0000001)
corels_rl.fit(x_train, y_train, features=features_unbias, prediction_name=prediction)
print(corels_rl.get_status())
train_acc = np.average(corels_rl.predict(x_train) == y_train)
test_acc = np.average(corels_rl.predict(x_test) == y_test)
print("train_acc= ", train_acc)
print("test_acc = ", test_acc)
corels_rl.distributional_overfit(x_train, x_test, y_train, y_test)

DP_smooth_rl = DpSmoothGreedyRLClassifier(min_support=min_support, max_length=max_length, verbosity=verbosity, max_card=max_card, allow_negations=True, epsilon = epsilon, noise = "Laplace", confidence=0.98)
DP_smooth_rl.fit(x_train, y_train, features=features_unbias, prediction_name=prediction)
DP_smooth_rl.distributional_overfit(x_train, x_test, y_train, y_test)

greedy_rl = GreedyRLClassifier(min_support=0.05, max_length=max_length, verbosity=verbosity, max_card=max_card, allow_negations=True)
greedy_rl.fit(x_train, y_train, features=features_unbias, prediction_name=prediction)
greedy_rl.distributional_overfit(x_train, x_test, y_train, y_test)

"""

def compute_overfit(model, max_card, max_length, dataset, runs):
    overfit = np.zeros((2,runs))
    vul = np.zeros(runs)
    acc = np.zeros(runs)

    for seed in range(runs):
        if model == "corels":
            rl_model = CorelsClassifier(n_iter=1000000, map_type="prefix", policy="bfs", verbosity=[], ablation=0, max_card=max_card, min_support=0.00, max_length=10000, c=0.0000001)
        elif model == "greedy":
            rl_model=GreedyRLClassifier(min_support=0.0, max_length=max_length, verbosity=verbosity, max_card=max_card, allow_negations=True, seed=seed)        
        else : rl_model= DpSmoothGreedyRLClassifier(min_support=min_support, max_length=max_length, verbosity=verbosity, max_card=max_card, allow_negations=True, epsilon = epsilon, noise = "Laplace", confidence=0.99, seed = seed)
        
        X, y, features, prediction = load_from_csv("data/%s.csv" %dataset)
        X_unbias,features_unbias = dp.clean_dataset(X,features, dataset)
        N = len(X_unbias)            
        x_train, y_train, x_test, y_test= dp.split_dataset(X_unbias, y, 0.70, seed =seed)
        rl_model.fit(x_train, y_train, features=features_unbias, prediction_name=prediction)
        a,b = rl_model.distributional_overfit(x_train, x_test, y_train, y_test, show=False)
        overfit[:,seed] =a
        vul [seed] = b
        acc[seed] = np.average(rl_model.predict(x_test) == y_test)
        
        
    print("{0} : dist overfit : {1} +/- {2} | overall vulnerability : {3} +/- {4} on dataset {5}".format(model, np.average(overfit,axis=1), np.var(overfit,axis=1),  np.average(vul), np.var(vul), dataset))
    print("{0} : accuracy : {1} +/- {2}".format(model, np.average(acc), np.var(acc)))

    









#compute_overfit("corels", max_card, max_length, dataset, runs)
compute_overfit("greedy", max_card, max_length, dataset, runs)
print("\n")
compute_overfit("smooth-greedy", max_card, max_length, dataset, runs)





