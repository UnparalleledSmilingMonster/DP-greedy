import itertools
from rocknrun import *



def command_exec(values:list, params:list, seed = False):
    assert len(values) == len(params)
    cmd = "python3 ../main.py"
    for i in range(len(params)):
        cmd += " --" + params[i] + " " + str(values[i])
    if seed :
        cmd += " --seed #SEED" 
    cmd += " --dataset #BENCHMARK"
    return cmd 
        

def gen_methods(array_params:list, params:list, name:str, seed = False):
    methods = []
    for (idx,values) in enumerate(itertools.product(*array_params)):
        methods.append("Method {0} {1}".format(idx,name))
        methods.append(command_exec(values, params, seed=seed))
    return str(len(methods)//2)+ " methods\n" + "\n".join(methods) +"\n"
 
def write_keyfile(filename:str, methods:list, benchmarks:list, seeds=None):
    with open(filename, "w") as f:
        f.write(methods)
        f.write(str(len(benchmarks)) + " benchmarks\n")
        for benchmark in benchmarks: 
            #Benchmark name = benchmark value
            f.write(benchmark + "\n")
            f.write(benchmark + "\n")
        if seeds is not None: f.write(" ".join([str(i) for i in range(seeds)]) +"\n")         
        f.close()
            

def setup():
    
    #Hyperparameters
    """
    max_lengths = [3, 5, 7]
    max_cards = [2]
    min_supports = [0.05, 0.10, 0.15, 0.20, 0.25]
    epsilons= [0.1, 0.5, 1, 5, 10, 50]
    deltas = [None]
    confidences = [0.95, 0.98]    
    """
    
    #For debug purposes :
    max_lengths = [5]
    max_cards = [2]
    min_supports = [0.20]
    epsilons= [1]
    deltas = [None]
    confidences = [0.98]   
    
    
    #Benchmarks
    benchmarks = ["compas", "adult"]
    
    #Seeds :
    #seeds = 100
    seeds = 1
   
    #Generate methods for Greedy RL vanilla : No seed because deterministic
    methods = gen_methods([max_lengths, max_cards, ["vanilla"]], ["max_length", "max_card", "mechanism"], "greedy") 
    write_keyfile("experiments_greedy.key", methods, benchmarks) 
    
    #Generate methods for regular DP:
    mechanisms_global = ["global-Laplace", "global-Gaussian", "Exponential"]
    methods = gen_methods([max_lengths, max_cards, epsilons, deltas, mechanisms_global],\
     ["max_length", "max_card", "epsilon", "delta", "mechanism"], "global", seed=True)
    write_keyfile("experiments_DP.key", methods, benchmarks, seeds)
  
    
    #Generate methods for smooth DP:
    mechanisms_smooth = ["smooth-Cauchy", "smooth-Laplace"]
    methods = gen_methods([max_lengths, max_cards, min_supports, epsilons, deltas, confidences, mechanisms_smooth],\
     ["max_length", "max_card", "min_support", "epsilon", "delta", "confidence", "mechanism"], "smooth", seed = True)
    write_keyfile("experiments_DP_smooth.key", methods, benchmarks, seeds)
  



if __name__ == '__main__' :
    setup()
    e = Experiment(["experiments_greedy", "experiments_DP", "experiments_DP_smooth"])
    e.generate_jobs(timeout='00:01:00')

