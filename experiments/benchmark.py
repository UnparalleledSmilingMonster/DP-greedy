from corels import load_from_csv, RuleList, CorelsClassifier
from HeuristicRL import GreedyRLClassifier
from HeuristicRL_DP import DPGreedyRLClassifier
from HeuristicRL_DP_smooth import DpSmoothGreedyRLClassifier
import numpy as np
import DP as dp

from prettytable import PrettyTable


#Eventually write to file
filename = "DP_results/benchmark.txt"

epsilon_letter='\u03B5'
delta_letter='\u03B4'
lambda_letter = '\u03BB'

verbosity=[]

#Benchmark Table :
t = PrettyTable(['dataset', 'Mechanism', epsilon_letter, delta_letter, lambda_letter, 'C_max', 'N', 'Runs', 'Accuracy'])

max_card = 2
epsilon = 1
llambda = 0.05
max_length = 5

#TODO: Add Laplace noise WITHOUT smooth sensitivity
#TODO: Test on more datasets
#TODO: change style of the table
#TODO: Finaaaaally : run for a LOT of runs (try the computation platform)
#TODO: Enjoy the result. And put it on latex :D

def benchmark(runs = 10, methods = ["smooth-Laplace", "smooth-Cauchy", "Exponential"], datasets = ["compas", "adult"]):
  
    for dataset in datasets:
        
        X, y, features, prediction = load_from_csv("data/%s.csv" %dataset)
        X_unbias,features_unbias = dp.clean_dataset(X,features, ["Race", "Age", "Gender"])
        N = len(X)
        
        #First compute the baseline algorithm (Greedy -Tree)
        greedy_rl = GreedyRLClassifier(min_support=llambda, max_length=max_length, verbosity=verbosity, max_card=max_card, allow_negations=True)
        greedy_rl.fit(X_unbias, y, features=features_unbias, prediction_name=prediction)        
        t.add_row([dataset, 'GreedyRL', 'x', 'x', llambda, max_card, N, runs,np.average(greedy_rl.predict(X_unbias) == y)])
        
        #DP mechanisms for Greedy RL : 
        for method in methods:
            if method.startswith("smooth"):
                noise = method.split("-")[1]
                res = np.zeros(runs)
                for i in range(runs):
                    DP_rl = DpSmoothGreedyRLClassifier(min_support=llambda, max_length=max_length, verbosity=verbosity, max_card=max_card, allow_negations=True, epsilon = epsilon, delta =1/(N**2) , noise = noise) #delta is 0 for Cauchy noise
                    DP_rl.fit(X_unbias, y, features=features_unbias, prediction_name=prediction)
                
                res[i]= np.average(DP_rl.predict(X_unbias) == y)
                t.add_row([dataset, method, epsilon, DP_rl.delta, llambda, max_card, N, runs,np.mean(res)])
           
           
            else : #exponential mechanism, if you add other mechanisms, do more if/else on the value of method
                res = np.zeros(runs)
                for i in range(runs):
                    DP_rl =  DPGreedyRLClassifier(min_support=llambda, max_length=max_length, verbosity=verbosity, max_card=max_card, allow_negations=True, epsilon = epsilon, delta = 0)
                    DP_rl.fit(X_unbias, y, features=features_unbias, prediction_name=prediction)
                
                res[i]= np.average(DP_rl.predict(X_unbias) == y)
                t.add_row([dataset, method, epsilon, DP_rl.delta, llambda, max_card, N, runs,np.mean(res)], divider = True)
                
                         
benchmark(1)
print(t)    
    
    
