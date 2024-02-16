import itertools
from rocknrun import *
import os
import numpy as np

counter = 0

def command_exec(array_params:list, params:list, dataset:str, directory = "./pfcalcul/results/"):
    global counter
    for (idx,values) in enumerate(itertools.product(*array_params)):
            cmd = "python3 main.py"
            #print(values)
            #print(idx)
            for i in range(len(params)):
                cmd += " --" + params[i] + " " + str(values[i])
            
            cmd += " --dataset " + dataset
            filename = directory +"expe_local_"+str(counter)+".out"
            os.system(cmd + " > " + filename) 
            counter +=1

    
 #Hyperparameters
 
max_lengths = [3, 5, 7]
max_cards = [2]
min_supports = [0.05, 0.10]
epsilons= [0.1, 0.5, 1, 5, 10, 50]
deltas = [None]
confidences = [0.99]    


#For visualization : 
"""
max_lengths=[5]
max_cards = [2]
min_supports = [0.05]
epsilons = np.logspace(-2, 2, num = 200)
deltas = [None]
confidences = [0.99]

"""
#Hyperparameters

max_lengths_german = [3,4,5]
max_cards_german = [1]
min_supports_german = [0.10, 0.15]



#For visualization : 
"""
max_lengths_german=[5]
max_cards_german = [1]
min_supports_german = [0.12]
"""

seed_nb = 100
seeds = [i for i in range(seed_nb)]


mechanisms_smooth = ["smooth-Cauchy", "smooth-Laplace"]
params_smooth = ["max_length", "max_card", "min_support", "epsilon", "delta", "confidence", "mechanism", "seed"]
val_smooth = [max_lengths, max_cards, min_supports, epsilons, deltas, confidences, mechanisms_smooth, seeds]

mechanisms_global = ["global-Laplace", "global-Gaussian", "global-Exponential"] #, "local-Exponential"]
params_dp = ["max_length", "max_card", "epsilon", "delta", "mechanism", "seed"]
val_dp =[max_lengths, max_cards, epsilons, deltas, mechanisms_global, seeds]

params_vanilla = ["max_length", "max_card", "mechanism", "seed"]
val_vanilla = [max_lengths, max_cards, ["vanilla"], seeds]

#Different params for german credit: 
val_smooth_german  = [max_lengths_german, max_cards_german, min_supports_german, epsilons, deltas, confidences, mechanisms_smooth, seeds]
val_dp_german =[max_lengths_german, max_cards_german, epsilons, deltas, mechanisms_global, seeds]
val_vanilla_german = [max_lengths_german, max_cards_german, ["vanilla"], seeds]

    
    
if __name__ == '__main__' :

    datasets = ["compas", "adult", "german_credit"]
        
    for dataset in ["compas", "adult"] :    
        command_exec(val_vanilla, params_vanilla, dataset)
        command_exec(val_dp, params_dp, dataset)
        command_exec(val_smooth, params_smooth, dataset)    
       
    dataset = "german_credit"
    command_exec(val_vanilla_german, params_vanilla, dataset)
    command_exec(val_dp_german, params_dp, dataset)
    command_exec(val_smooth_german, params_smooth, dataset)    
     

