import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd
from matplotlib.lines import Line2D

figures_folder = 'Figures'

# Slurm task parallelism
datasets = ["compas"]
methods = ["DL8.5", "sklearn_DT"] # 0 for CORELS, 1 for DL8.5, 2 for sklearn DT (CART)   

# MPI parallelism
random_seeds = [i for i in range(5)] # for 1) data train/test split and 2) methods initialization
min_support_params = [0.01*i for i in range(1,6)] # minimum proportion of training examples that a rule (or a leaf) must capture
max_depth_params = [i for i in range(1,11)]

# Plotting parameters
linestyles = {"DL8.5": 'solid', "sklearn_DT": 'dotted'}
depthcolors = {}

# Random generation of colors
import random
from random import randint
random.seed(42)
colors_list = []
n = len(max_depth_params) + len(min_support_params)
for i in range(n):
    colors_list.append('#%06X' % randint(0, 0xFFFFFF))
i = 0
for max_depth in max_depth_params:
    depthcolors[max_depth] = colors_list[i]
    i += 1

for min_support in min_support_params:
    depthcolors[min_support] = colors_list[i]
    i += 1

res_dict = dict()

for dataset in datasets:
    res_dict[dataset] = dict()

    for method in methods:
        res_dict[dataset][method] = dict()

        # Find per dataset-method results file
        fileName = './results_graham/%s_%s.csv' %(method, dataset) #_proportions
        try:
            res = pd.read_csv(fileName)
        except:
            print("File not found: ", fileName)
            exit()

        # Iterate over results
        for index, row in res.iterrows():
            # Double check
            assert(row['method'] == method)
            assert(row['dataset'] == dataset)

            # Read parameters
            random_state_value = row['random_state_value']
            min_support = row['min_support']
            max_depth = row['max_depth']
        
            # Read results
            duration = row['duration']
            train_acc = row['train_acc']
            test_acc = row['test_acc']
            n_elementary_tokens = row['n_elementary_tokens']
            n_branches_rules = row['n_branches_rules']
            average_tokens_per_examples = row['average_tokens_per_examples']
            n_elementary_tokens_path = row['n_elementary_tokens_path']
            entropy_reduction_ratio = row['entropy_reduction_ratio']

            # Save them
            if not max_depth in res_dict[dataset][method].keys():
                res_dict[dataset][method][max_depth] = dict()

            if not min_support in res_dict[dataset][method][max_depth].keys():
                res_dict[dataset][method][max_depth][min_support] = [entropy_reduction_ratio]
            else:
                res_dict[dataset][method][max_depth][min_support].append(entropy_reduction_ratio)

        # Average over the folds

        for max_depth in res_dict[dataset][method].keys():
            for min_support in res_dict[dataset][method][max_depth].keys():
                assert(len(res_dict[dataset][method][max_depth][min_support]) == 5) # double check
                res_dict[dataset][method][max_depth][min_support] = np.average(res_dict[dataset][method][max_depth][min_support])
                
        for max_depth in res_dict[dataset][method].keys():
            min_supports_list = res_dict[dataset][method][max_depth].keys()
            entropy_reduction_list = [res_dict[dataset][method][max_depth][min_support] for min_support in min_supports_list]
            plt.plot(min_supports_list, entropy_reduction_list, linestyle=linestyles[method], marker='o', c=depthcolors[max_depth]) #, label='max depth %d' %max_depth)
    
    plt.xlabel('Min. leaf support')
    plt.ylabel('$\mathsf{Dist}_G(\mathcal{W}_{\mathcal{DT}},\mathcal{D}_{orig})$') #  \newmetric(\generalizedpdataset_{\interpretablemodel},\pdataset_{orig})

    plt.savefig("%s/%s_per_max_depth.pdf" %(figures_folder, dataset), bbox_inches='tight')
    plt.clf()

    legendFig = plt.figure("Legend plot")
    legend_elements = []
    for method in methods:
        legend_elements.append(Line2D([0], [0], color='black', linestyle=linestyles[method], label=method))
    for max_depth in max_depth_params:
        legend_elements.append(Line2D([0], [0], marker='o', color=depthcolors[max_depth], lw=1, linestyle='None', label='max depth %d' %max_depth))
    legendFig.legend(handles=legend_elements, loc='center', ncol=4)
    legendFig.savefig("%s/%s_per_max_depth_legend.pdf" %(figures_folder, dataset), bbox_inches='tight')
    plt.clf()


    for method in methods:
        for min_support in min_supports_list:
            max_depth_list = res_dict[dataset][method].keys()
            entropy_reduction_list = [res_dict[dataset][method][max_depth][min_support] for max_depth in max_depth_list]
            plt.plot(max_depth_list, entropy_reduction_list, linestyle=linestyles[method], marker='o', c=depthcolors[min_support]) #, label='min support %.2f' %min_support)
    
    plt.xlabel('Max. depth')
    plt.ylabel('$\mathsf{Dist}_G(\mathcal{W}_{\mathcal{DT}},\mathcal{D}_{orig})$') #  \newmetric(\generalizedpdataset_{\interpretablemodel},\pdataset_{orig})
    plt.savefig("%s/%s_per_min_support.pdf" %(figures_folder, dataset), bbox_inches='tight')
    plt.clf()

    legendFig = plt.figure("Legend plot")
    legend_elements = []
    for method in methods:
        legend_elements.append(Line2D([0], [0], color='black', linestyle=linestyles[method], label=method))
    for min_support in min_support_params:
        legend_elements.append(Line2D([0], [0], marker='o', color=depthcolors[min_support], lw=1, linestyle='None', label='min support %.2f' %min_support))
    legendFig.legend(handles=legend_elements, loc='center', ncol=4)
    legendFig.savefig("%s/%s_per_min_support_legend.pdf" %(figures_folder, dataset), bbox_inches='tight')
    plt.clf()