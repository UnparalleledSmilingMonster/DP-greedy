import numpy as np
import matplotlib.pyplot as plt


import seaborn
import json
seaborn.set_theme()


epsilon_letter='\u03B5'
delta_letter='\u03B4'
lambda_letter = '\u03BB'


def method_epsilon_graph(res, dataset = "compas", max_length = 5):
    methods = sorted(set([res[key][2] for key in res]))
    if "vanilla" in methods : methods.remove("vanilla")
    epsilons = set([res[key][3] for key in res])
    if "x" in epsilons : epsilons.remove('x')
    epsilons = sorted(epsilons)
    
    accuracies = np.zeros((len(methods), len(epsilons)))
    var = np.zeros((len(methods), len(epsilons)))
       
    
    for key in res :
        if res[key][1] != max_length or res[key][0] != dataset : continue
        if res[key][2] == "vanilla":
            best = res[key][10]
            continue
            
        else:
            i = methods.index(res[key][2])
            j = epsilons.index(res[key][3])        
            accuracies[i][j] = res[key][10]
            var[i][j] = res[key][12]

    
    plt.figure(figsize=(12,10))
    for i in range(len(methods)):    
        plt.plot(epsilons, accuracies[i], label = methods[i])
        plt.fill_between(epsilons, accuracies[i] - var[i], accuracies[i] + var[i])
    plt.axhline(y = best,linestyle = ':', linewidth=3, label = "Vanilla baseline")
    plt.xlabel(epsilon_letter, fontsize = 23)
    plt.ylabel("Accuracy", fontsize = 23)
    plt.title("Method comparison for dataset {0}".format(dataset))
    plt.xscale("log")
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08),
          fancybox=True, shadow=True, ncol=6,fontsize=13)
    plt.show()    
        
            
    

if __name__ == '__main__':    
    params = ['dataset', 'Mechanism', epsilon_letter, delta_letter, lambda_letter, "Confidence", 'C_max', 'N', 'Runs', 'Avg. Time(s)', 'Accuracy']
    res={}
    with open("summary_visu.nfo", 'r') as summary :
        res = json.load(summary)
        summary.close()
    method_epsilon_graph(res, dataset = "german_credit")
    
   
    






