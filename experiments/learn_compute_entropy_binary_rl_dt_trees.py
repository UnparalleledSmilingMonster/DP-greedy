from pydl85 import DL85Classifier
from corels import *
import time
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np 
from expes_utils import compute_entropy_single_example
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import argparse
from local_config import ccanada_expes
if ccanada_expes:
    from mpi4py import MPI


parser = argparse.ArgumentParser(description='Probabilistic dataset reconstruction from interpretable model experiments')
parser.add_argument('--expe_id', type=int, default=0, choices= [0,1,2,3], help='method-dataset combination (for now, only COMPAS and tic-tac-toe supported)')
args = parser.parse_args()

if ccanada_expes:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    verbosity = -1 # >= 0 minimal infos >=2 basic script infos >=3 CORELS infos >= 5 basic info about recursive computations >= 10 detailed info about recursive computations
else:
    rank = 9
    verbosity = 2 # >= 0 minimal infos >=2 basic script infos >=3 CORELS infos >= 5 basic info about recursive computations >= 10 detailed info about recursive computations
    
# Script parameters
test_size_ratio = 0.2 # only to check generalization
plot_extension = "pdf"
max_time = 3600 # seconds

# Slurm task parallelism
expe_id=args.expe_id
datasets = ["compas", "tic-tac-toe"]
methods = ["DL8.5", "sklearn_DT"] # 0 for CORELS, 1 for DL8.5, 2 for sklearn DT (CART)    
slurm_expes = []
for d in datasets:
    for m in methods:
        slurm_expes.append([d, m])

dataset = slurm_expes[expe_id][0] # "tic-tac-toe" # "compas"
method = slurm_expes[expe_id][1] # "DL8.5" # "sklearn_DT"

if verbosity >= 0:
    print("Slurm #expes = ", len(slurm_expes))
    print("Current expe: dataset %s, method %s" %(dataset, method))

# MPI parallelism
random_seeds = [i for i in range(5)] # for 1) data train/test split and 2) methods initialization
min_support_params = [0.01*i for i in range(2,6)] # minimum proportion of training examples that a rule (or a leaf) must capture
max_depth_params = [i for i in range(1,11)]

if dataset == "tic-tac-toe":
    min_support_params.append(0.005)
    min_support_params.append(0.001)

configs_list = []
for rs in random_seeds: # 5 values
    for msp in min_support_params: # 5 values
        for mdp in max_depth_params: # 10 values
            configs_list.append([rs, msp, mdp])

random_state_value = configs_list[rank][0]
min_support = configs_list[rank][1]
max_depth = configs_list[rank][2]

if verbosity >= 0:
    print("MPI #params = ", len(configs_list))
    print("Current expe: random_state_value %d, min_support %.3f, max_depth %d" %(random_state_value, min_support, max_depth))

# Load the data
if dataset in ["compas", 'adult']:
    X, y, features, prediction = load_from_csv("data/%s.csv" %dataset)
else: # datasets from the pydl8.5 repository
    data_full = np.genfromtxt("data/%s.txt" %dataset, delimiter=' ')
    X = data_full[:, 1:]
    y = data_full[:, 0]
    X = X.astype('int32')
    y = y.astype('int32')
    features, prediction = "na", "na"

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_ratio, random_state=random_state_value)
total_features = X_train.shape[1]
n_samples = y_train.size
if verbosity >= 0:
    print("Dataset shape: ", X.shape, "prediction is ", prediction)
    print("Training set shape: ", X_train.shape)
    print("-----------------------------------------------------------------")

X = "shouldnotbeusedanymore"
y = "shouldnotbeusedanymore"
trees_min_sample_leaves = max([1,int(min_support * n_samples)])

# NO-KNOWLEDGE DATASET computations
single_example_possibilities = 2**(total_features) #num_possibilities([]) #2**X.shape[1] # Binary features
single_example_entropy = compute_entropy_single_example(single_example_possibilities) 
no_knowledge_dataset_entropy = n_samples * single_example_entropy
if verbosity >= 0:
    print("Number of possible reconstructions: %d for %d examples." %(single_example_possibilities, n_samples))
    print("Original dataset entropy (with no knowledge) = ", n_samples, "*", single_example_entropy, " = ", no_knowledge_dataset_entropy)
    print("-----------------------------------------------------------------")

n_elementary_tokens = 0
n_branches_rules = 0
average_tokens_per_examples = 0
n_elementary_tokens_path = 0

if method == "DL8.5":
    # return the error and the majority class
    def error(sup_iter):
        supports = list(sup_iter)
        maxindex = np.argmax(supports)
        return sum(supports) - supports[maxindex], maxindex
    
    def num_possibilities(splits_list):
        splits_indexes = np.unique(splits_list)
        if splits_indexes.size != len(splits_list): # means that a split is done twice in the same branch -> should not happen
            raise ValueError("splits_indexes.size = " + str(splits_indexes.size) + " != " +  "len(splits_list) = " + str(len(splits_list)))
        leaf_possibilities = 2**(total_features - len(splits_list))
        return leaf_possibilities

    def print_tree(tree, offset='', preset=''):
        # impression couchée
        if 'error' in tree: # leaf
            print("%s %s Leaf: class %d (support %d)" %(offset, preset, tree['value'], tree['support']))
        else: # internal node (at least one child)
            if 'right' in tree:
                print_tree(tree['right'], offset=offset+'                 ', preset='(0)')
            print('%s %s Node: feat. %d' %(offset, preset, tree['feat']))
            if 'left' in tree:
                print_tree(tree['left'], offset=offset+'                 ', preset='(1)')

    def complete_tree(tree, splits_list=[]):
        '''
        Adds the support of each leaf (already approximately given via error and probas but I want the exact (not rounded) value)
        '''
        if 'error' in tree: # leaf
            # Manually compute how many training samples are captured by the leaf
            capt = np.ones(shape=(X_train.shape[0]))
            for a_feat in splits_list:
                if a_feat > 0:
                    a_feat -= 1
                    capt = np.logical_and(capt, X_train[:,a_feat])
                elif a_feat < 0:
                    a_feat = abs(a_feat)
                    a_feat -= 1
                    capt = np.logical_and(capt, np.logical_not(X_train[:,a_feat]))
                else:
                    raise ValueError("a_feat = " + str(a_feat) + " < 0")
            leaf_support = np.sum(capt)
            tree["support"] = leaf_support
            # Check if OK with DL8.5 computed error at leaf
            capt_pos = np.sum(np.logical_and(capt, y_train))
            if tree['value'] == 0:
                leaf_error = capt_pos
            elif tree['value'] == 1:
                leaf_error = leaf_support - capt_pos
            else:
                raise ValueError('leaf value ' + str(err_value) + " unexpected")
            assert(leaf_error == tree['error'])
        else: # internal node (at least one child)
            new_splits_list_left = splits_list+ [1+tree['feat']]
            new_splits_list_right = splits_list+ [-(1+tree['feat'])]
            if 'left' in tree:
                complete_tree(tree['left'], splits_list=new_splits_list_left)
            if 'right' in tree:
                complete_tree(tree['right'], splits_list=new_splits_list_right)
    
    leaves_possibilities_list = []
    leaves_entropy_list = []
    leaves_support = []
    leaves_single_example_entropy_list = []

    def explore_tree(tree, splits_list=[]): # Completes leaves_possibilities_list and leaves_entropy_list
        global n_elementary_tokens, leaves_possibilities_list, leaves_entropy_list, leaves_support, n_branches_rules, n_elementary_tokens_path, average_tokens_per_examples
        if 'error' in tree: # leaf
            n_branches_rules += 1
            leaf_possibilities = num_possibilities(splits_list)
            #if verbosity >= 2:
            #    print("Leaf: %d possibilities." %leaf_possibilities)
            leaves_possibilities_list.append(leaf_possibilities)
            leaf_entropy = compute_entropy_single_example(leaf_possibilities)
            leaves_entropy_list.append(leaf_entropy * tree['support'])
            leaves_support.append(tree['support'])
            average_tokens_per_examples += tree['support'] * len(splits_list)
            n_elementary_tokens_path += len(splits_list)
            leaves_single_example_entropy_list.append(leaf_entropy)
        else: # internal node (at least one child)
            n_elementary_tokens += 1
            new_splits_list_left = splits_list+ [tree['feat']]
            new_splits_list_right = splits_list+ [tree['feat']]
            if 'left' in tree:
                explore_tree(tree['left'], splits_list=new_splits_list_left)
            if 'right' in tree:
                explore_tree(tree['right'], splits_list=new_splits_list_right)
                
    clf = DL85Classifier(max_depth=max_depth, min_sup = trees_min_sample_leaves, fast_error_function=error, time_limit=max_time)
    start = time.perf_counter()
    clf.fit(X_train, y_train)
    duration = time.perf_counter() - start
    
    train_acc = np.mean(y_train == clf.predict(X_train))
    test_acc = np.mean(y_test == clf.predict(X_test))
    complete_tree(clf.tree_) # self-made procedure to explicitly compute, check and append the support of each leaf (already contained via #errors and probas but rounded)
    if verbosity >= 0:
        print("Model built. Duration of building =", round(duration, 4))
        #print("Accuracy DL8.5 on training set =", round(clf.accuracy_, 4))
        print("Accuracy DL8.5 on training set =", round(train_acc, 4))
        
        print_tree(clf.tree_)

        print("-----------------------------------------------------------------")

    explore_tree(clf.tree_)

    assert(sum(leaves_support) == n_samples) # better double checking
    average_tokens_per_examples /= n_samples
    if verbosity >= 2:
        print("# possible worlds per rule: ", leaves_possibilities_list)
        print("total entropy per rule: ", leaves_entropy_list)


    reconstructed_dataset_entropy = sum(leaves_entropy_list)
    entropy_reduction_ratio = reconstructed_dataset_entropy/no_knowledge_dataset_entropy
    
    # print the tree
    import graphviz
    dot = clf.export_graphviz()
    graph = graphviz.Source(dot, format=plot_extension)
    graph.render("./models/DL8.5_tree_%s_%d_%.2f_%d" %(dataset, max_depth, min_support, random_state_value))

    res = [[dataset, method, random_state_value, min_support, max_depth, duration, train_acc, test_acc, reconstructed_dataset_entropy, no_knowledge_dataset_entropy, n_elementary_tokens, n_branches_rules, entropy_reduction_ratio, average_tokens_per_examples, n_elementary_tokens_path, leaves_support, leaves_single_example_entropy_list]]

elif method == "sklearn_DT":

    clf = DecisionTreeClassifier(random_state=1+random_state_value, max_depth=max_depth, min_samples_leaf=trees_min_sample_leaves)

    start = time.perf_counter()
    clf.fit(X_train, y_train)
    duration = time.perf_counter() - start

    train_acc = clf.score(X_train, y_train)
    test_acc = clf.score(X_test, y_test)

    if verbosity >= 0:
        print("Model built. Duration of building =", round(duration, 4))
        print("Accuracy CART (sklearn) on training set =", round(train_acc, 4))
        
        print("-----------------------------------------------------------------")

    n_nodes = clf.tree_.node_count
    children_left = clf.tree_.children_left # For all nodes in the tree, list of their left children (or -1 for leaves)
    children_right = clf.tree_.children_right # For all nodes in the tree, list of their right children (or -1 for leaves)
    nodes_features = clf.tree_.feature # For all nodes in the tree, list of their used feature (or -2 for leaves)
    nodes_threshold = clf.tree_.threshold # For all nodes in the tree, list of their used threshold (or -2. for leaves)
    nodes_impurity = clf.tree_.impurity # For all nodes in the tree, list of their impurity
    nodes_value = clf.tree_.value # For all nodes in the tree, list of their value (support for both classes)

    def retrieve_branches(number_nodes, children_left_list, children_right_list):
        """Retrieve decision tree branches"""
        
        # Calculate if a node is a leaf
        is_leaves_list = [(False if cl != cr else True) for cl, cr in zip(children_left_list, children_right_list)]
        
        # Store the branches paths
        paths = []
        
        for i in range(number_nodes):
            if is_leaves_list[i]:
                # Search leaf node in previous paths
                end_node = [path[-1] for path in paths]

                # If it is a leave node yield the path
                if i in end_node:
                    output = paths.pop(np.argwhere(i == np.array(end_node))[0][0])
                    yield output

            else:
                
                # Origin and end nodes
                origin, end_l, end_r = i, children_left_list[i], children_right_list[i]

                # Iterate over previous paths to add nodes
                for index, path in enumerate(paths):
                    if origin == path[-1]:
                        paths[index] = path + [end_l]
                        paths.append(path + [end_r])

                # Initialize path in first iteration
                if i == 0:
                    paths.append([i, children_left[i]])
                    paths.append([i, children_right[i]])

    all_branches = list(retrieve_branches(n_nodes, children_left, children_right))

    def num_possibilities(splits_list):
        splits_indexes = np.unique(splits_list)
        if splits_indexes.size != len(splits_list): # means that a split is done twice in the same branch -> should not happen
            raise ValueError("splits_indexes.size = " + str(splits_indexes.size) + " != " +  "len(splits_list) = " + str(len(splits_list)))
        leaf_possibilities = 2**(total_features - len(splits_list))
        return leaf_possibilities
        
    leaves_possibilities_list = []
    leaves_entropy_list = []
    leaves_single_example_entropy_list = []
    leaves_support = []

    for a_possible_split in nodes_features: # Count internal nodes
        if a_possible_split != -2:
            n_elementary_tokens += 1
    
    for a_branch in all_branches: # 
        n_branches_rules += 1
        splits_list = []
        leaf_support = -1
        for a_node_id in a_branch:
            node_feat = nodes_features[a_node_id]
            if node_feat == -2: # leaf 
                leaf_support = np.sum(nodes_value[a_node_id])
            else:
                splits_list.append(node_feat)
        if verbosity >= 5:
            print('Leaf support ', leaf_support, ' splits = ', splits_list)
        assert(leaf_support > 0) # ensure leaf was found in the current branch
        leaf_possibilities = num_possibilities(splits_list)
        leaves_possibilities_list.append(leaf_possibilities)
        leaf_entropy = compute_entropy_single_example(leaf_possibilities)
        leaves_entropy_list.append(leaf_entropy * leaf_support)
        leaves_single_example_entropy_list.append(leaf_entropy)
        leaves_support.append(leaf_support)
        average_tokens_per_examples += (leaf_support * len(splits_list))
        n_elementary_tokens_path += len(splits_list)

    assert(sum(leaves_support) == n_samples) # better double checking
    average_tokens_per_examples /= n_samples

    if verbosity >= 2:
        print("# possible worlds per rule: ", leaves_possibilities_list)
        print("total entropy per rule: ", leaves_entropy_list)

    reconstructed_dataset_entropy = sum(leaves_entropy_list)
    entropy_reduction_ratio = reconstructed_dataset_entropy/no_knowledge_dataset_entropy

    from sklearn.tree import plot_tree
    import matplotlib.pyplot as plt
    plot_tree(clf, filled=True)#, feature_names = features)
    plt.savefig("./models/sklearn_tree_%s_%d_%.2f_%d.%s" %(dataset, max_depth, min_support, random_state_value, plot_extension), bbox_inches='tight')
    plt.clf()

    res = [[dataset, method, random_state_value, min_support, max_depth, duration, train_acc, test_acc, reconstructed_dataset_entropy, no_knowledge_dataset_entropy, n_elementary_tokens, n_branches_rules, entropy_reduction_ratio, average_tokens_per_examples, n_elementary_tokens_path, leaves_support, leaves_single_example_entropy_list]]

else:
    raise ValueError("Unknown method " + str(method))

'''
sorted_leaves_support = np.asarray([x for _, x in sorted(zip(leaves_single_example_entropy_list, leaves_support))])
sorted_leaves_single_example_entropy_list = np.sort(leaves_single_example_entropy_list)
sorted_leaves_support = np.cumsum(sorted_leaves_support)


import matplotlib.pyplot as plt
plt.plot(sorted_leaves_support, sorted_leaves_single_example_entropy_list)

plt.savefig("./results/tree_cdf_entropy_%s.%s" %(dataset, plot_extension), bbox_inches='tight')
'''

# Gather the results for the 5 folds on process 0
if ccanada_expes:
    res = comm.gather(res, root=0)

if rank == 0 or not ccanada_expes:
    # save results
    fileName = './results/%s_%s.csv' %(method, dataset) #_proportions
    import csv
    with open(fileName, mode='w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(['dataset', 'method', 'random_state_value', 'min_support', 'max_depth', 'duration', 'train_acc', 'test_acc', 'reconstructed_dataset_entropy', 'no_knowledge_dataset_entropy', 'n_elementary_tokens', 'n_branches_rules', 'entropy_reduction_ratio', 'average_tokens_per_examples', 'n_elementary_tokens_path', 'leaves_support', 'sorted_leaves_single_example_entropy_list'])
        for i in range(len(res)):
            if ccanada_expes:
                for j in range(len(res[i])):
                    csv_writer.writerow(res[i][j])
            else:
                csv_writer.writerow(res[i])

if verbosity >= 0:
    print("Reconstructed dataset joint entropy (with no knowledge) = ", reconstructed_dataset_entropy)
    print("Distance to actual (deterministic) dataset (for %d elementary tokens, and %d branches or rules) = " %(n_elementary_tokens, n_branches_rules), entropy_reduction_ratio)

# Save results in a unified format