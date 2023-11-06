from corels import load_from_csv, RuleList, CorelsClassifier
from HeuristicRL import GreedyRLClassifier
from HeuristicRL_DP import DPGreedyRLClassifier
from HeuristicRL_DP_smooth import DpSmoothGreedyRLClassifier
from HeuristicRL_DP_noise import DpNoiseGreedyRLClassifier

import numpy as np
import DP as dp


import time
from tqdm import tqdm
from prettytable import PrettyTable
from rich.progress import Progress, BarColumn, TimeElapsedColumn

#Eventually write to file
filename = "DP_results/benchmark.txt"

epsilon_letter='\u03B5'
delta_letter='\u03B4'
lambda_letter = '\u03BB'

verbosity=[]

#Benchmark Table :
t = PrettyTable(['dataset', 'Mechanism', epsilon_letter, delta_letter, lambda_letter, "Confidence", 'C_max', 'N', 'Runs', 'Avg. Time(s)', 'Accuracy'])

max_card = 2
epsilons = [0.1, 1 , 10]
llambda = 0.05
max_length = 5
confidence = 0.98

#TODO: Add Laplace noise WITHOUT smooth sensitivity
#TODO: Test on more datasets
#TODO: change style of the table
#TODO: Finaaaaally : run for a LOT of runs (try the computation platform)
#TODO: Enjoy the result. And put it on latex :D

def benchmark(runs = 10, methods = ["smooth-Laplace", "smooth-Cauchy", "Laplace", "Gaussian", "Exponential"], datasets = ["compas", "adult"]):
  
  
    with Progress(*Progress.get_default_columns(), TimeElapsedColumn()) as progress:
        dataset_bar = progress.add_task("[red]Processing...", total=len(datasets))        
        
        buffer = 0
        for dataset in datasets:
            progress.update(dataset_bar, description = "[red]Processing {0}".format(dataset))                       
            
            X, y, features, prediction = load_from_csv("data/%s.csv" %dataset)
            X_unbias,features_unbias = dp.clean_dataset(X,features, dp.get_biases(dataset))
            N = len(X)
            
            
            method_bar = progress.add_task("[cyan]Method...", total=len(methods)*len(epsilons)+1)
            progress.update(method_bar, description = "[cyan]Method « GreedyRL »")
            start= time.time()
            #First compute the baseline algorithm (Greedy -Tree)
            greedy_rl = GreedyRLClassifier(min_support=llambda, max_length=max_length, verbosity=verbosity, max_card=max_card, allow_negations=True)
            greedy_rl.fit(X_unbias, y, features=features_unbias, prediction_name=prediction)        
            t.add_row([dataset, 'GreedyRL', 'x', 'x', llambda, 'x', max_card, N, runs,pretty_format(time.time() - start), pretty_format(np.average(greedy_rl.predict(X_unbias) == y))])
            progress.update(method_bar, advance = 1)
            
            
            #DP mechanisms for Greedy RL : 
            for method in methods:
                progress.update(method_bar, description = "[cyan]Method « {0} »".format(method))
                
                for epsilon in epsilons :
                
                    if method.startswith("smooth"):
                        start= time.time()
                        noise = method.split("-")[1]
                        res = np.zeros(runs)
                        for i in range(runs):
                            DP_rl = DpSmoothGreedyRLClassifier(min_support=llambda, max_length=max_length, verbosity=verbosity, max_card=max_card, allow_negations=True, epsilon = epsilon, delta =1/(N**2) ,confidence = confidence, noise = noise) #delta is 0 for Cauchy noise
                            DP_rl.fit(X_unbias, y, features=features_unbias, prediction_name=prediction)                    
                            res[i]= np.average(DP_rl.predict(X_unbias) == y)
                            
                        t.add_row([dataset, method, epsilon, pretty_format(DP_rl.delta, 'e', 2), llambda, confidence, max_card, N, runs, pretty_format((time.time() - start)/runs), pretty_format(np.mean(res))])
                        progress.update(method_bar, advance = 1)
                   
                   
                    elif method == "Exponential":
                        start= time.time()
                        res = np.zeros(runs)
                        for i in range(runs):
                            DP_rl =  DPGreedyRLClassifier(min_support=llambda, max_length=max_length, verbosity=verbosity, max_card=max_card, allow_negations=True, epsilon = epsilon, delta = 0)
                            DP_rl.fit(X_unbias, y, features=features_unbias, prediction_name=prediction)                    
                            res[i]= np.average(DP_rl.predict(X_unbias) == y)
                            
                        t.add_row([dataset, method, epsilon, DP_rl.delta, llambda, max_card, 'x', N, runs, pretty_format((time.time() - start)/runs), pretty_format(np.mean(res))], divider = True if epsilon == epsilons[-1] else False)
                        progress.update(method_bar, advance = 1)
                        
                    else : #Global sensitivity Laplace and Gaussian
                        start= time.time()
                        res = np.zeros(runs)
                        for i in range(runs):
                            DP_rl =  DpNoiseGreedyRLClassifier(min_support=llambda, max_length=max_length, verbosity=verbosity, max_card=max_card, allow_negations=True, epsilon = epsilon, noise = method)
                            DP_rl.fit(X_unbias, y, features=features_unbias, prediction_name=prediction)                    
                            res[i]= np.average(DP_rl.predict(X_unbias) == y)
                            
                        t.add_row([dataset, method, epsilon, pretty_format(DP_rl.delta, "e", 2), llambda, 'x', max_card, N, runs, pretty_format((time.time() - start)/runs), pretty_format(np.mean(res))])
                        progress.update(method_bar, advance = 1)

            progress.update(method_bar, description= "[green]All methods were benchmarked for {0}".format(dataset))
            
            buffer +=1
            string = "[red]Processing {0}".format(dataset)
            dataset_bar = swap_bars(progress, dataset_bar, buffer, "[red]Processing {0}".format(dataset), len(datasets))
           
        progress.update(dataset_bar, description = "[green]All datasets were processed")
       


def swap_bars(prog, bar, buffer, desc, tot):
    prog.remove_task(bar)
    bar = prog.add_task(desc, total=tot)        
    prog.update(bar, advance = buffer)
    return bar
    

def pretty_format(number, mode = "f", num = 3):
    if number == 0 : return number  #Just so that 0 keeps being 0 (when we print delta for pure DP)
    
    form = "{" + "0:.{0}{1}".format(num,mode) + "}"
    return form.format(number)
    
                         
benchmark(runs = 20, datasets = ["compas", "adult"]) #, "australian-credit"])
print(t) 

with open(filename, 'w') as w:
    w.write(t.get_string())

    
