from corels import load_from_csv, RuleList, CorelsClassifier
from HeuristicRL import GreedyRLClassifier
from HeuristicRL_DP import DPGreedyRLClassifier
from HeuristicRL_DP_smooth import DpSmoothGreedyRLClassifier
from DP_global_old import DpNoiseGreedyRLClassifier

import numpy as np
import DP as dp
import time

import argparse

epsilon_letter='\u03B5'
delta_letter='\u03B4'
lambda_letter = '\u03BB'


confidence = 0.99
max_card = 2

def pformat(var, mode ="f", num=3 ):
    if (var is None or var =="None" or var==0) : return 'x'
    form = "{" + "0:.{0}{1}".format(num,mode) + "}"
    return form.format(float(var))
    
def rformat(var):
    return 'x' if (var is None or var==0) else var
    

if __name__ == '__main__':    
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_card", type=int, help="The maximum of conditions in a rule", default=2, required=False)
    parser.add_argument("--min_support", type=float, help="The minimum ratio of elements to be split", default=0.0, required = False)
    parser.add_argument("--epsilon", type =float, help="The " + epsilon_letter + " budget of the DP", default=0.0, required=False)
    parser.add_argument("--delta", help="The " + delta_letter + " budget of the DP", default=None, required=False)
    parser.add_argument("--confidence", type=float, help="The confidence value for the rule to respect the minimum support", default=0.0, required=False)
    parser.add_argument("--seed", type=int, help="The seed for replicability", default =42, required = False)
    parser.add_argument("--test_train", type=float, help="The train/test ratio (precise train)", default=0.8, required = False)

    parser.add_argument("--dataset", type=str, help="The dataset to train the model on", required=True)
    parser.add_argument("--max_length", type=int, help="The maximum length of a rule list", required=True)
    parser.add_argument("--mechanism", type=str, help="The mechanism used for the model. Choose between [vanilla, smooth-Laplace, smooth-Cauchy, global-Laplace, global-Gaussian, Exponential]", required=True)

    args = parser.parse_args()

    X, y, features, prediction = load_from_csv("data/%s.csv" %args.dataset)
    X_unbias,features_unbias = dp.clean_dataset(X,features, dp.get_biases(args.dataset))
    N = len(X_unbias)

    assert(args.test_train >0)
    X_unbias_train, y_train, X_unbias_test, y_test= dp.split_dataset(X_unbias, y, args.test_train, seed = args.seed)

    if args.mechanism == "vanilla":
        start= time.time()
        greedy_rl = GreedyRLClassifier(min_support=0.0, max_length=args.max_length, max_card=args.max_card, allow_negations=True, seed = args.seed )
        greedy_rl.fit(X_unbias_train, y_train, features=features_unbias, prediction_name=prediction) 
        end=time.time() - start        
        
    elif args.mechanism.endswith("Exponential"):
        start= time.time()
        sens = args.mechanism.split("-")[0]
        greedy_rl =  DPGreedyRLClassifier(min_support=0.0, max_length=args.max_length, max_card=args.max_card, allow_negations=True, epsilon = args.epsilon, delta = 0.0, sensitivity = sens, seed = args.seed)
        greedy_rl.fit(X_unbias_train, y_train, features=features_unbias, prediction_name=prediction)       
        end=time.time() - start                        
        
    elif args.mechanism.startswith("smooth"):
        start= time.time()
        noise = args.mechanism.split("-")[1]
        greedy_rl = DpSmoothGreedyRLClassifier(min_support=args.min_support, max_length=args.max_length, max_card=args.max_card, allow_negations=True, epsilon = args.epsilon, delta =args.delta, confidence = confidence, noise = noise, seed = args.seed)
        greedy_rl.fit(X_unbias_train, y_train, features=features_unbias, prediction_name=prediction)        
        args.delta =greedy_rl.delta       
        end=time.time() - start           

    elif args.mechanism.startswith("global"):
        start= time.time()
        noise = args.mechanism.split("-")[1]
        greedy_rl = DpNoiseGreedyRLClassifier(min_support=0.0, max_length=args.max_length, max_card=args.max_card, allow_negations=True, epsilon = args.epsilon, delta =args.delta, noise = noise, seed = args.seed) 
        greedy_rl.fit(X_unbias_train, y_train, features=features_unbias, prediction_name=prediction)  
        args.delta =greedy_rl.delta
        end=time.time() - start           
             
    else : raise Exception("The mechanism desired is not implemented.")
    #print(greedy_rl)
    print([args.dataset, args.max_length, args.mechanism, rformat(args.epsilon), pformat(args.delta, "e", 2), rformat(args.min_support), N, end, np.average(greedy_rl.predict(X_unbias_train) == y_train), np.average(greedy_rl.predict(X_unbias_test) == y_test)] )



