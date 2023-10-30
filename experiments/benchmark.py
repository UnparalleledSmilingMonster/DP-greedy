from corels import load_from_csv, RuleList, CorelsClassifier
from HeuristicRL import GreedyRLClassifier
from HeuristicRL_DP import DPGreedyRLClassifier
from HeuristicRL_DP_smooth import DpSmoothGreedyRLClassifier
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
t = PrettyTable(['dataset', 'Mechanism', epsilon_letter, delta_letter, lambda_letter, 'C_max', 'N', 'Runs', 'Avg. Time(s)', 'Accuracy'])

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
  
  
    with Progress(*Progress.get_default_columns(), TimeElapsedColumn()) as progress:
        dataset_bar = progress.add_task("[red]Processing...", total=len(datasets))        
        
        buffer = 0
        for dataset in datasets:
            progress.update(dataset_bar, description = "[red]Processing {0}".format(dataset))                       
            
            X, y, features, prediction = load_from_csv("data/%s.csv" %dataset)
            X_unbias,features_unbias = dp.clean_dataset(X,features, ["Race", "Age", "Gender"])
            N = len(X)
            
            
            method_bar = progress.add_task("[cyan]Method...", total=len(methods)+1)
            progress.update(method_bar, description = "[cyan]Method « GreedyRL »")
            start= time.time()
            #First compute the baseline algorithm (Greedy -Tree)
            greedy_rl = GreedyRLClassifier(min_support=llambda, max_length=max_length, verbosity=verbosity, max_card=max_card, allow_negations=True)
            greedy_rl.fit(X_unbias, y, features=features_unbias, prediction_name=prediction)        
            t.add_row([dataset, 'GreedyRL', 'x', 'x', llambda, max_card, N, runs,pretty_format(time.time() - start), pretty_format(np.average(greedy_rl.predict(X_unbias) == y))])
            progress.update(method_bar, advance = 1)
            
            
            #DP mechanisms for Greedy RL : 
            for method in methods:
                progress.update(method_bar, description = "[cyan]Method « {0} »".format(method))
                
                if method.startswith("smooth"):
                    start= time.time()
                    noise = method.split("-")[1]
                    res = np.zeros(runs)
                    for i in range(runs):
                        DP_rl = DpSmoothGreedyRLClassifier(min_support=llambda, max_length=max_length, verbosity=verbosity, max_card=max_card, allow_negations=True, epsilon = epsilon, delta =1/(N**2) , noise = noise) #delta is 0 for Cauchy noise
                        DP_rl.fit(X_unbias, y, features=features_unbias, prediction_name=prediction)                    
                        res[i]= np.average(DP_rl.predict(X_unbias) == y)
                        
                    t.add_row([dataset, method, epsilon, pretty_format(DP_rl.delta, 'e', 2), llambda, max_card, N, runs, pretty_format((time.time() - start)/runs), pretty_format(np.mean(res))])
                    progress.update(method_bar, advance = 1)
               
               
                else : #exponential mechanism, if you add other mechanisms, do more if/else on the value of method
                    start= time.time()
                    res = np.zeros(runs)
                    for i in range(runs):
                        DP_rl =  DPGreedyRLClassifier(min_support=llambda, max_length=max_length, verbosity=verbosity, max_card=max_card, allow_negations=True, epsilon = epsilon, delta = 0)
                        DP_rl.fit(X_unbias, y, features=features_unbias, prediction_name=prediction)                    
                        res[i]= np.average(DP_rl.predict(X_unbias) == y)
                        
                    t.add_row([dataset, method, epsilon, DP_rl.delta, llambda, max_card, N, runs, pretty_format((time.time() - start)/runs), pretty_format(np.mean(res))], divider = True)
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
    
                         
benchmark(runs = 1, datasets = ["compas", "adult"]) #, "australian-credit"])
print(t)    
    
    
