import numpy as np
import matplotlib.pyplot as plt
import seaborn
import json
seaborn.set_theme()


epsilon_letter='\u03B5'
delta_letter='\u03B4'
lambda_letter = '\u03BB'


def method_epsilon_graph(res, dataset = "compas", max_length = 7):
    methods = sorted(set([res[key][2] for key in res]))
    methods.remove("vanilla")
    epsilons = set([res[key][3] for key in res])
    epsilons.remove('x')
    epsilons = sorted(epsilons)
    
    accuracies = np.zeros((len(methods), len(epsilons)))
    
    for key in res :
        if res[key][1] != max_length or res[key][0] != dataset : continue
        if res[key][2] == "vanilla":
            best = res[key][11]
            continue
            
        else:
            i = methods.index(res[key][2])
            j = epsilons.index(res[key][3])        
            accuracies[i][j] = res[key][11]
    
    plt.figure(figsize=(12,10))
    for i in range(len(methods)):    
        plt.plot(epsilons, accuracies[i], label = methods[i])
    plt.axhline(y = best,linestyle = ':', linewidth=3, label = "Vanilla baseline")
    plt.xlabel(epsilon_letter)
    plt.ylabel("Accuracy")
    plt.title("Method comparison for dataset {0}".format(dataset))
    plt.legend()          
    plt.show()    
        
            
    

if __name__ == '__main__':    
    params = ['dataset', 'Mechanism', epsilon_letter, delta_letter, lambda_letter, "Confidence", 'C_max', 'N', 'Runs', 'Avg. Time(s)', 'Accuracy']
    res={}
    with open("summary.nfo", 'r') as summary :
        res = json.load(summary)
        summary.close()
    method_epsilon_graph(res, dataset = "adult")
    
   
    






