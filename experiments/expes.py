import itertools
from rocknrun import *



def command_exec(values:list, params:list, dataset:str, file:str, seed = False):
    assert len(values) == len(params)
    cmd = "python3 main.py"
    for i in range(len(params)):
        cmd += " --" + params[i] + " " + str(values[i])
    if seed :
        cmd += " --seed #SEED" 
    cmd += " --dataset" + dataset
    return cmd 
        

    
 #Hyperparameters
""" 
max_lengths = [3, 5, 7]
max_cards = [2]
min_supports = [0.05, 0.10]
epsilons= [0.1, 0.5, 1, 10, 50]
deltas = [None]
confidences = [0.99]    



#For visualization : 

max_lengths=[5]
max_cards = [2]
min_supports = [0.05]
epsilons = np.logspace(-2, 2, num = 200)
deltas = [None]
confidences = [0.99]
 """
   
#For debug purposes :
max_lengths = [5]
max_cards = [2]
min_supports = [0.10]
epsilons= [1,10]
deltas = [None]
confidences = [0.98]   

#Seeds :
#seeds = 100
seeds = 10





#Hyperparameters
""" 
max_lengths = [3,4,5]
max_cards = [1]
min_supports = [0.10, 0.15]
epsilons= [0.1, 0.5, 1, 10, 50]
deltas = [None]
confidences = [0.99]    



#For visualization : 
"""
max_lengths_german=[5]
max_cards_german = [1]
min_supports_german = [0.12]
epsilons_german = np.logspace(-2, 2, num = 200)
deltas_german = [None]
confidences_german = [0.99]
 
"""
#For debug purposes :
max_lengths = [5]
max_cards = [2]
min_supports = [0.20]
epsilons= [1]
deltas = [None]
confidences = [0.98]   
"""

    

mechanisms_smooth = ["smooth-Cauchy", "smooth-Laplace"]
params_smooth = ["max_length", "max_card", "min_support", "epsilon", "delta", "confidence", "mechanism"]
val_smooth  = [max_lengths, max_cards, min_supports, epsilons, deltas, confidences, mechanisms_smooth],

mechanisms_global = ["global-Laplace", "global-Gaussian", "Exponential"]
params_dp = ["max_length", "max_card", "epsilon", "delta", "mechanism"]
val_dp =[max_lengths, max_cards, epsilons, deltas, mechanisms_global

params_vanilla = ["max_length", "max_card", "mechanism"]
val_vanilla = [max_lengths, max_cards, ["vanilla"]]






if __name__ == '__main__' :

    datasets = ["compas", "adult", "german_credit"]
    for dataset in datasets :
    
    def gen_methods(array_params:list, params:list, name:str, seed = False):
        for (idx,values) in enumerate(itertools.product(*array_params)):
            methods.append(command_exec(values, params, seed=seed))
        return str(len(methods)//2)+ " methods\n" + "\n".join(methods) +"\n"
        if dataset== "german_credit":
            command_exec(val_vanilla, params_vanilla)
    setup()
    e = Experiment(["experiments_greedy", "experiments_DP", "experiments_DP_smooth"])
    e.generate_jobs(timeout='00:01:00')

