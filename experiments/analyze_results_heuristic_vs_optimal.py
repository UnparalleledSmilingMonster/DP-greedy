import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd
from matplotlib.lines import Line2D

figures_folder = 'Figures'
extension = "png"
# Slurm task parallelism
datasets = ["compas"]
methods = ["DL8.5", "sklearn_DT"] # 0 for CORELS, 1 for DL8.5, 2 for sklearn DT (CART)   
criteria = ['entropy_reduction_ratio', 'n_elementary_tokens', 'n_branches_rules', 'average_tokens_per_examples', 'n_elementary_tokens_path', 'test_acc', 'train_acc']

# MPI parallelism
random_seeds = [i for i in range(5)] # for 1) data train/test split and 2) methods initialization
min_support_params = [0.01*i for i in range(1,6)] # minimum proportion of training examples that a rule (or a leaf) must capture
max_depth_params = [i for i in range(1,11)]

# Plotting parameters
linestyles = {"DL8.5": 'solid', "sklearn_DT": 'dotted'}
markers = {"DL8.5": '*', "sklearn_DT": 'o'}
markersizes = {"DL8.5": None, "sklearn_DT": '4'}

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
            if row['method'] == "CORELS" and int(row['max_width']) != 1:
                continue
            # Read parameters
            random_state_value = row['random_state_value']
            min_support = row['min_support']
            max_depth = row['max_depth']
        
            # Read results
            '''duration = row['duration']
            train_acc = row['train_acc']
            test_acc = row['test_acc']
            n_elementary_tokens = row['n_elementary_tokens']
            n_branches_rules = row['n_branches_rules']
            average_tokens_per_examples = row['average_tokens_per_examples']
            n_elementary_tokens_path = row['n_elementary_tokens_path']
            entropy_reduction_ratio = row['entropy_reduction_ratio']'''
            
            # Save them
            if not max_depth in res_dict[dataset][method].keys():
                res_dict[dataset][method][max_depth] = dict()

            if not min_support in res_dict[dataset][method][max_depth].keys():
                res_dict[dataset][method][max_depth][min_support] = dict()
                for criterion in criteria:
                    res_dict[dataset][method][max_depth][min_support][criterion] = [row[criterion]]
            else:
                for criterion in criteria:
                    res_dict[dataset][method][max_depth][min_support][criterion].append(row[criterion])

        # Average over the folds

        for max_depth in res_dict[dataset][method].keys():
            for min_support in res_dict[dataset][method][max_depth].keys():
                for criterion in criteria:
                    assert(len(res_dict[dataset][method][max_depth][min_support][criterion]) == 5) # double check
                    res_dict[dataset][method][max_depth][min_support][criterion] = np.average(res_dict[dataset][method][max_depth][min_support][criterion])
    # Plot entropy_reduction_ratio = f(min_support)
    for method in methods:            
        for max_depth in res_dict[dataset][method].keys():
            min_supports_list = res_dict[dataset][method][max_depth].keys()
            entropy_reduction_list = [res_dict[dataset][method][max_depth][min_support]['entropy_reduction_ratio'] for min_support in min_supports_list]
            plt.plot(min_supports_list, entropy_reduction_list, linestyle=linestyles[method], marker=markers[method], markersize=markersizes[method], c=depthcolors[max_depth]) #, label='max depth %d' %max_depth)
    
    plt.xlabel('Min. leaf support')
    plt.ylabel('$\mathsf{Dist}_G(\mathcal{W}_{\mathcal{DT}},\mathcal{D}_{orig})$') #  \newmetric(\generalizedpdataset_{\interpretablemodel},\pdataset_{orig})

    plot_name = 'entropy_reduction_ratio_f_min_support'
    plt.savefig("%s/%s_%s.%s" %(figures_folder, plot_name, dataset, extension), bbox_inches='tight')
    plt.clf()

    legendFig = plt.figure("Legend plot")
    legend_elements = []
    for method in methods:
        legend_elements.append(Line2D([0], [0], color='black', linestyle=linestyles[method], label=method, marker=markers[method], markersize=markersizes[method]))
    for max_depth in max_depth_params:
        legend_elements.append(Line2D([0], [0], markersize=markersizes[method], marker='o', color=depthcolors[max_depth], lw=1, linestyle='None', label='max depth %d' %max_depth))
    legendFig.legend(handles=legend_elements, loc='center', ncol=4)
    legendFig.savefig("%s/%s_%s_legend.%s" %(figures_folder, plot_name, dataset, extension), bbox_inches='tight')
    plt.clf()

    # Plot entropy_reduction_ratio = f(max_depth)
    for method in methods:
        for min_support in min_supports_list:
            max_depth_list = res_dict[dataset][method].keys()
            entropy_reduction_list = [res_dict[dataset][method][max_depth][min_support]['entropy_reduction_ratio'] for max_depth in max_depth_list]
            plt.plot(max_depth_list, entropy_reduction_list, linestyle=linestyles[method], marker=markers[method], markersize=markersizes[method], c=depthcolors[min_support]) #, label='min support %.2f' %min_support)
    
    plt.xlabel('Max. depth')
    plt.ylabel('$\mathsf{Dist}_G(\mathcal{W}_{\mathcal{DT}},\mathcal{D}_{orig})$') #  \newmetric(\generalizedpdataset_{\interpretablemodel},\pdataset_{orig})
    
    plot_name = 'entropy_reduction_ratio_f_max_depth'
    plt.savefig("%s/%s_%s.%s" %(figures_folder, plot_name, dataset, extension), bbox_inches='tight')
    plt.clf()

    legendFig = plt.figure("Legend plot")
    legend_elements = []
    for method in methods:
        legend_elements.append(Line2D([0], [0], color='black', linestyle=linestyles[method], label=method, marker=markers[method], markersize=markersizes[method]))
    for min_support in min_support_params:
        legend_elements.append(Line2D([0], [0], markersize=markersizes[method], marker='o', color=depthcolors[min_support], lw=1, linestyle='None', label='min support %.2f' %min_support))
    legendFig.legend(handles=legend_elements, loc='center', ncol=4)
    legendFig.savefig("%s/%s_%s_legend.%s" %(figures_folder, plot_name, dataset, extension), bbox_inches='tight')
    plt.clf()

    # Plot entropy_reduction_ratio = f(n_elementary_tokens)
    for method in methods:
        for min_support in min_supports_list:
            max_depth_list = res_dict[dataset][method].keys()
            entropy_reduction_list = [res_dict[dataset][method][max_depth][min_support]['entropy_reduction_ratio'] for max_depth in max_depth_list]
            n_elementary_tokens_list = [res_dict[dataset][method][max_depth][min_support]['n_elementary_tokens'] for max_depth in max_depth_list]
            n_elementary_tokens_list, entropy_reduction_list = zip(*sorted(zip(n_elementary_tokens_list, entropy_reduction_list))) # To order
            plt.plot(n_elementary_tokens_list, entropy_reduction_list, linestyle=linestyles[method], marker=markers[method], markersize=markersizes[method], c=depthcolors[min_support]) #, label='min support %.2f' %min_support)
    
    plt.xlabel('# elementary tokens')
    plt.ylabel('$\mathsf{Dist}_G(\mathcal{W}_{\mathcal{DT}},\mathcal{D}_{orig})$') #  \newmetric(\generalizedpdataset_{\interpretablemodel},\pdataset_{orig})
    
    plot_name = 'entropy_reduction_ratio_f_n_elementary_tokens'
    plt.savefig("%s/%s_%s.%s" %(figures_folder, plot_name, dataset, extension), bbox_inches='tight')
    plt.clf()

    legendFig = plt.figure("Legend plot")
    legend_elements = []
    for method in methods:
        legend_elements.append(Line2D([0], [0], color='black', linestyle=linestyles[method], label=method, marker=markers[method], markersize=markersizes[method]))
    for min_support in min_support_params:
        legend_elements.append(Line2D([0], [0], markersize=markersizes[method], marker='o', color=depthcolors[min_support], lw=1, linestyle='None', label='min support %.2f' %min_support))
    legendFig.legend(handles=legend_elements, loc='center', ncol=4)
    legendFig.savefig("%s/%s_%s_legend.%s" %(figures_folder, plot_name, dataset, extension), bbox_inches='tight')
    plt.clf()

    # Plot average_tokens_per_examples = f(max_depth)
    for method in methods:
        for min_support in min_supports_list:
            max_depth_list = res_dict[dataset][method].keys()
            average_tokens_per_examples_list = [res_dict[dataset][method][max_depth][min_support]['average_tokens_per_examples'] for max_depth in max_depth_list]
            max_depth_list, average_tokens_per_examples_list  = zip(*sorted(zip(max_depth_list, average_tokens_per_examples_list))) # To order
            plt.plot(max_depth_list, average_tokens_per_examples_list, linestyle=linestyles[method], marker=markers[method], markersize=markersizes[method], c=depthcolors[min_support]) #, label='min support %.2f' %min_support)
    
    plt.ylabel('average # tokens per examples')
    plt.xlabel('Max. depth') #  \newmetric(\generalizedpdataset_{\interpretablemodel},\pdataset_{orig})
    
    plot_name = 'average_tokens_per_examples_f_max_depth'
    plt.savefig("%s/%s_%s.%s" %(figures_folder, plot_name, dataset, extension), bbox_inches='tight')
    plt.clf()

    legendFig = plt.figure("Legend plot")
    legend_elements = []
    for method in methods:
        legend_elements.append(Line2D([0], [0], color='black', linestyle=linestyles[method], label=method, marker=markers[method], markersize=markersizes[method]))
    for min_support in min_support_params:
        legend_elements.append(Line2D([0], [0], markersize=markersizes[method], marker='o', color=depthcolors[min_support], lw=1, linestyle='None', label='min support %.2f' %min_support))
    legendFig.legend(handles=legend_elements, loc='center', ncol=4)
    legendFig.savefig("%s/%s_%s_legend.%s" %(figures_folder, plot_name, dataset, extension), bbox_inches='tight')
    plt.clf()

    # Plot n_elementary_tokens = f(max_depth)
    for method in methods:
        for min_support in min_supports_list:
            max_depth_list = res_dict[dataset][method].keys()
            n_elementary_tokens_list = [res_dict[dataset][method][max_depth][min_support]['n_elementary_tokens'] for max_depth in max_depth_list]
            max_depth_list, n_elementary_tokens_list  = zip(*sorted(zip(max_depth_list, n_elementary_tokens_list))) # To order
            plt.plot(max_depth_list, n_elementary_tokens_list, linestyle=linestyles[method], marker=markers[method], markersize=markersizes[method], c=depthcolors[min_support]) #, label='min support %.2f' %min_support)
    
    plt.ylabel('# elementary tokens')
    plt.xlabel('Max. depth') #  \newmetric(\generalizedpdataset_{\interpretablemodel},\pdataset_{orig})
    
    plot_name = 'n_elementary_tokens_f_max_depth'
    plt.savefig("%s/%s_%s.%s" %(figures_folder, plot_name, dataset, extension), bbox_inches='tight')
    plt.clf()

    legendFig = plt.figure("Legend plot")
    legend_elements = []
    for method in methods:
        legend_elements.append(Line2D([0], [0], color='black', linestyle=linestyles[method], label=method, marker=markers[method], markersize=markersizes[method]))
    for min_support in min_support_params:
        legend_elements.append(Line2D([0], [0], markersize=markersizes[method], marker='o', color=depthcolors[min_support], lw=1, linestyle='None', label='min support %.2f' %min_support))
    legendFig.legend(handles=legend_elements, loc='center', ncol=4)
    legendFig.savefig("%s/%s_%s_legend.%s" %(figures_folder, plot_name, dataset, extension), bbox_inches='tight')
    plt.clf()

    # Plot n_branches_rules = f(max_depth)
    for method in methods:
        for min_support in min_supports_list:
            max_depth_list = res_dict[dataset][method].keys()
            n_branches_rules_list = [res_dict[dataset][method][max_depth][min_support]['n_branches_rules'] for max_depth in max_depth_list]
            max_depth_list, n_branches_rules_list  = zip(*sorted(zip(max_depth_list, n_branches_rules_list))) # To order
            plt.plot(max_depth_list, n_branches_rules_list, linestyle=linestyles[method], marker=markers[method], markersize=markersizes[method], c=depthcolors[min_support]) #, label='min support %.2f' %min_support)
    
    plt.ylabel('# branches')
    plt.xlabel('Max. depth') #  \newmetric(\generalizedpdataset_{\interpretablemodel},\pdataset_{orig})
    
    plot_name = 'n_branches_f_max_depth'
    plt.savefig("%s/%s_%s.%s" %(figures_folder, plot_name, dataset, extension), bbox_inches='tight')
    plt.clf()

    legendFig = plt.figure("Legend plot")
    legend_elements = []
    for method in methods:
        legend_elements.append(Line2D([0], [0], color='black', linestyle=linestyles[method], label=method, marker=markers[method], markersize=markersizes[method]))
    for min_support in min_support_params:
        legend_elements.append(Line2D([0], [0], markersize=markersizes[method], marker='o', color=depthcolors[min_support], lw=1, linestyle='None', label='min support %.2f' %min_support))
    legendFig.legend(handles=legend_elements, loc='center', ncol=4)
    legendFig.savefig("%s/%s_%s_legend.%s" %(figures_folder, plot_name, dataset, extension), bbox_inches='tight')
    plt.clf()

    # Plot entropy_reduction_ratio = f(train_acc)
    for method in methods:
        for min_support in min_supports_list:
            max_depth_list = res_dict[dataset][method].keys()
            entropy_reduction_list = [res_dict[dataset][method][max_depth][min_support]['entropy_reduction_ratio'] for max_depth in max_depth_list]
            train_acc_list = [res_dict[dataset][method][max_depth][min_support]['train_acc'] for max_depth in max_depth_list]
            train_acc_list, entropy_reduction_list = zip(*sorted(zip(train_acc_list, entropy_reduction_list))) # To order
            plt.plot(train_acc_list, entropy_reduction_list, linestyle=linestyles[method], marker=markers[method], markersize=markersizes[method], c=depthcolors[min_support]) #, label='min support %.2f' %min_support)
    
    plt.xlabel('Training accuracy')
    plt.ylabel('$\mathsf{Dist}_G(\mathcal{W}_{\mathcal{DT}},\mathcal{D}_{orig})$') #  \newmetric(\generalizedpdataset_{\interpretablemodel},\pdataset_{orig})
    
    plot_name = 'entropy_reduction_ratio_f_train_acc_list'
    plt.savefig("%s/%s_%s.%s" %(figures_folder, plot_name, dataset, extension), bbox_inches='tight')
    plt.clf()

    legendFig = plt.figure("Legend plot")
    legend_elements = []
    for method in methods:
        legend_elements.append(Line2D([0], [0], color='black', linestyle=linestyles[method], label=method, marker=markers[method], markersize=markersizes[method]))
    for min_support in min_support_params:
        legend_elements.append(Line2D([0], [0], markersize=markersizes[method], marker='o', color=depthcolors[min_support], lw=1, linestyle='None', label='min support %.2f' %min_support))
    legendFig.legend(handles=legend_elements, loc='center', ncol=4)
    legendFig.savefig("%s/%s_%s_legend.%s" %(figures_folder, plot_name, dataset, extension), bbox_inches='tight')
    plt.clf()

    # Plot entropy_reduction_ratio = f(test_acc)
    for method in methods:
        for min_support in min_supports_list:
            max_depth_list = res_dict[dataset][method].keys()
            entropy_reduction_list = [res_dict[dataset][method][max_depth][min_support]['entropy_reduction_ratio'] for max_depth in max_depth_list]
            test_acc_list = [res_dict[dataset][method][max_depth][min_support]['test_acc'] for max_depth in max_depth_list]
            test_acc_list, entropy_reduction_list = zip(*sorted(zip(test_acc_list, entropy_reduction_list))) # To order
            plt.plot(test_acc_list, entropy_reduction_list, linestyle=linestyles[method], marker=markers[method], markersize=markersizes[method], c=depthcolors[min_support]) #, label='min support %.2f' %min_support)
    
    plt.xlabel('Test accuracy')
    plt.ylabel('$\mathsf{Dist}_G(\mathcal{W}_{\mathcal{DT}},\mathcal{D}_{orig})$') #  \newmetric(\generalizedpdataset_{\interpretablemodel},\pdataset_{orig})
    
    plot_name = 'entropy_reduction_ratio_f_test_acc'
    plt.savefig("%s/%s_%s.%s" %(figures_folder, plot_name, dataset, extension), bbox_inches='tight')
    plt.clf()

    legendFig = plt.figure("Legend plot")
    legend_elements = []
    for method in methods:
        legend_elements.append(Line2D([0], [0], color='black', linestyle=linestyles[method], label=method, marker=markers[method], markersize=markersizes[method]))
    for min_support in min_support_params:
        legend_elements.append(Line2D([0], [0], markersize=markersizes[method], marker='o', color=depthcolors[min_support], lw=1, linestyle='None', label='min support %.2f' %min_support))
    legendFig.legend(handles=legend_elements, loc='center', ncol=4)
    legendFig.savefig("%s/%s_%s_legend.%s" %(figures_folder, plot_name, dataset, extension), bbox_inches='tight')
    plt.clf()

    # Plot train_acc = f(n_elementary_tokens)
    for method in methods:
        for min_support in min_supports_list:
            max_depth_list = res_dict[dataset][method].keys()
            train_acc_list = [res_dict[dataset][method][max_depth][min_support]['train_acc'] for max_depth in max_depth_list]
            n_elementary_tokens_list = [res_dict[dataset][method][max_depth][min_support]['n_elementary_tokens'] for max_depth in max_depth_list]
            n_elementary_tokens_list, train_acc_list = zip(*sorted(zip(n_elementary_tokens_list, train_acc_list))) # To order
            plt.plot(n_elementary_tokens_list, train_acc_list, linestyle=linestyles[method], marker=markers[method], markersize=markersizes[method], c=depthcolors[min_support]) #, label='min support %.2f' %min_support)
    
    plt.ylabel('Training accuracy')
    plt.xlabel('# elementary tokens') #  \newmetric(\generalizedpdataset_{\interpretablemodel},\pdataset_{orig})
    
    plot_name = 'train_acc_f_n_elementary_tokens'
    plt.savefig("%s/%s_%s.%s" %(figures_folder, plot_name, dataset, extension), bbox_inches='tight')
    plt.clf()

    legendFig = plt.figure("Legend plot")
    legend_elements = []
    for method in methods:
        legend_elements.append(Line2D([0], [0], color='black', linestyle=linestyles[method], label=method, marker=markers[method], markersize=markersizes[method]))
    for min_support in min_support_params:
        legend_elements.append(Line2D([0], [0], markersize=markersizes[method], marker='o', color=depthcolors[min_support], lw=1, linestyle='None', label='min support %.2f' %min_support))
    legendFig.legend(handles=legend_elements, loc='center', ncol=4)
    legendFig.savefig("%s/%s_%s_legend.%s" %(figures_folder, plot_name, dataset, extension), bbox_inches='tight')
    plt.clf()

    # Plot test_acc = f(n_elementary_tokens)
    for method in methods:
        for min_support in min_supports_list:
            max_depth_list = res_dict[dataset][method].keys()
            test_acc_list = [res_dict[dataset][method][max_depth][min_support]['test_acc'] for max_depth in max_depth_list]
            n_elementary_tokens_list = [res_dict[dataset][method][max_depth][min_support]['n_elementary_tokens'] for max_depth in max_depth_list]
            n_elementary_tokens_list, test_acc_list = zip(*sorted(zip(n_elementary_tokens_list, test_acc_list))) # To order
            plt.plot(n_elementary_tokens_list, test_acc_list, linestyle=linestyles[method], marker=markers[method], markersize=markersizes[method], c=depthcolors[min_support]) #, label='min support %.2f' %min_support)
    
    plt.ylabel('Test accuracy')
    plt.xlabel('# elementary tokens') #  \newmetric(\generalizedpdataset_{\interpretablemodel},\pdataset_{orig})
    
    plot_name = 'test_acc_f_n_elementary_tokens'
    plt.savefig("%s/%s_%s.%s" %(figures_folder, plot_name, dataset, extension), bbox_inches='tight')
    plt.clf()

    legendFig = plt.figure("Legend plot")
    legend_elements = []
    for method in methods:
        legend_elements.append(Line2D([0], [0], color='black', linestyle=linestyles[method], label=method, marker=markers[method], markersize=markersizes[method]))
    for min_support in min_support_params:
        legend_elements.append(Line2D([0], [0], markersize=markersizes[method], marker='o', color=depthcolors[min_support], lw=1, linestyle='None', label='min support %.2f' %min_support))
    legendFig.legend(handles=legend_elements, loc='center', ncol=4)
    legendFig.savefig("%s/%s_%s_legend.%s" %(figures_folder, plot_name, dataset, extension), bbox_inches='tight')
    plt.clf()