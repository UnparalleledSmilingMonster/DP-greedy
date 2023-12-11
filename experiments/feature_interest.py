from corels import load_from_csv, RuleList, CorelsClassifier
from HeuristicRL import GreedyRLClassifier
from HeuristicRL_DP import DPGreedyRLClassifier
from HeuristicRL_DP_smooth import DpSmoothGreedyRLClassifier
import numpy as np
import DP as dp
import pandas as pd

def get_feature(features, i):
    if not features or abs(i) > len(features):
        return ""

    if i < 0:
        return "not " + features[-i - 1]
    else:
        return features[i - 1]
        
        

dataset = "adult"
min_support = 0.05
max_length = 7
max_card = 2
epsilon = 0.1
verbosity = [] # ["mine"] # ["mine"]
X, y, features, prediction = load_from_csv("data/%s.csv" %dataset)
seed = 35
X_unbias,features_unbias = dp.clean_dataset(X,features, dataset)
N = len(X_unbias)            
x_train, y_train, x_test, y_test= dp.split_dataset(X_unbias, y, 0.70, seed =seed)


def features_corr(dataset, top = 10):

    df = pd.read_csv("data/{0}.csv".format(dataset))
    corr_matrix = df.corr(method="pearson")
    target_feature = df.columns[-1]
    return list(np.abs(corr_matrix[target_feature]).sort_values(ascending=False).index[1:top])
     
features_of_interest = features_corr(dataset)
print("Features of interest:\n", features_of_interest)

def ratio_interest(model, features):
    dic ={}
    normalize = 0
    for feature in features : dic[feature] = 0
    rules = model.rl_.rules
    for i in range(len(rules)-1):
        rule_i = []
        for j in range(len(rules[i]["antecedents"])):
            rule_i.append(get_feature(model.rl_.features, rules[i]["antecedents"][j]))
        for feature in rule_i:
            feature= feature.replace("not ", "")
            if feature in dic : 
                dic[feature] += 1
    return sum(dic.values())/(len(rules)-1)
    
    

corels_rl = CorelsClassifier(n_iter=1000000, map_type="prefix", policy="lower_bound", verbosity=["rulelist"], ablation=0, max_card=max_card, min_support=0.05, max_length=7, c=0.0000001)
corels_rl.fit(x_train, y_train, features=features_unbias, prediction_name=prediction)
print(corels_rl.get_status())
print("Features of interest used:", ratio_interest(corels_rl, features_of_interest))


DP_smooth_rl = DpSmoothGreedyRLClassifier(min_support=min_support, max_length=max_length, verbosity=verbosity, max_card=max_card, allow_negations=True, epsilon = epsilon, noise = "Laplace", confidence=0.98)
DP_smooth_rl.fit(x_train, y_train, features=features_unbias, prediction_name=prediction)
print(DP_smooth_rl)
print("Features of interest used:", ratio_interest(DP_smooth_rl, features_of_interest))

greedy_rl = GreedyRLClassifier(min_support=0.05, max_length=max_length, verbosity=verbosity, max_card=max_card, allow_negations=True)
greedy_rl.fit(x_train, y_train, features=features_unbias, prediction_name=prediction)
print(greedy_rl)
print("Features of interest used:", ratio_interest(greedy_rl, features_of_interest))
