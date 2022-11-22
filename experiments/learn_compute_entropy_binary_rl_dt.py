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

parser = argparse.ArgumentParser(description='Probabilistic dataset reconstruction from interpretable model experiments')
parser.add_argument('--method_id', type=int, default=0, choices= [0,1,2], help='0 for CORELS, 1 for DL8.5, 2 for sklearn DT (CART)')
args = parser.parse_args()

# Script parameters
verbosity = 2 # >= 0 minimal infos >=2 basic script infos >=3 CORELS infos >= 5 basic info about recursive computations >= 10 detailed info about recursive computations
method_id=args.method_id
methods = ["CORELS", "DL8.5", "sklearn_DT"]
method = methods[method_id]
dataset = "compas"
test_size_ratio = 0.2 # only to check generalization
random_state_value = 42
plot_extension = "pdf"

# Load the data
X, y, features, prediction = load_from_csv("data/%s.csv" %dataset)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_ratio, random_state=random_state_value)
X = "shouldnotbeusedanymore"
y = "shouldnotbeusedanymore"
total_features = X_train.shape[1]
n_samples = y_train.size
if verbosity >= 0:
    print("Dataset shape: ", X_train.shape)
    print("-----------------------------------------------------------------")

# NO-KNOWLEDGE DATASET computations
single_example_possibilities = 2**(total_features) #num_possibilities([]) #2**X.shape[1] # Binary features
single_example_entropy = compute_entropy_single_example(single_example_possibilities) 
no_knowledge_dataset_entropy = n_samples * single_example_entropy
if verbosity >= 0:
    print("Number of possible reconstructions: %d for %d examples." %(single_example_possibilities, n_samples))
    print("Original dataset entropy (with no knowledge) = ", no_knowledge_dataset_entropy)
    print("-----------------------------------------------------------------")

max_time = 180 # seconds
min_support_param = 0.05 # proportion
trees_min_sample_leaves = int(min_support_param * n_samples)
maximum_depth = 4

n_elementary_tokens = 0
n_branches_rules = 0

if method == "CORELS":
    # CORELS parameters
   
    max_memory = 8000 # megabytes
    n_iter_param = 10 ** 9
    maximum_width = 2
    policy_param = 'lower_bound'
    # greater than zero to help pruning
    # but smaller than 1/n_samples because we don't want to trade-off accuracy
    cValue =  0.99*(1/n_samples) #n_examples_to_improve_f_obj/X.shape[0]

    corels_verbosity = []
    if verbosity >= 3:
        corels_verbosity.extend(['progress', 'mine'])
        
    def num_possibilities(antecedents_indexes):
        # a rule index is negative if it is a negation
        # else positive
        unique_antecedents_indexes = np.unique(antecedents_indexes)
        if 0 in unique_antecedents_indexes:
            raise ValueError("Should never happen (0 in unique_antecedents_indexes), exiting.")
        for k in unique_antecedents_indexes:
            if -k in unique_antecedents_indexes:
                return 0 # no possible world
        return 2**(total_features - unique_antecedents_indexes.size)

    def capt_i_j(i, j_list, all_rules):
        '''
        Number of worlds compatible with rules in j_list (conjunction) but actually captured by rule i in rule list all_rules
        '''
        rule_i_antecedents = all_rules[i]['antecedents']
        rules_in_j_antecedents = np.concatenate([all_rules[k]['antecedents'] for k in j_list])
        num_i_and_j = num_possibilities(np.concatenate([rule_i_antecedents, rules_in_j_antecedents]))
        if verbosity >= 10:
            print("num_%d_and_" %(i), j_list, ' = ', num_i_and_j)
        res = num_i_and_j - sum([capt_i_j(k, np.unique(np.concatenate([j_list, [i]])), all_rules) for k in range(0, i)])
        if verbosity >= 10:
            print("capt_%d_"%i, j_list, res)
        return res 

    def capt_rl(j, all_rules):
        '''
        Number of possible worlds for examples captured by rule j in rule list all_rules.
        '''
        res = capt_i_j(j, [j], all_rules)
        if verbosity >= 5:
            print("capt_rl(%d) = " %j, res, "\n")
        return res


    clf = CorelsClassifier(c=cValue, verbosity=corels_verbosity, policy=policy_param, n_iter=n_iter_param, max_card=maximum_width, max_length=maximum_depth, min_support=min_support_param) #, min_support=0.20)
    start = time.perf_counter()
    clf.fit(X_train, y_train, features=features, time_limit=max_time, memory_limit=max_memory)
    duration = time.perf_counter() - start
    acc = clf.score(X_train, y_train)

    if verbosity >= 0:
        print("Model built. Duration of building =", round(duration, 4))
        print("Status = ", clf.get_status())
        print((clf.rl_))
        print("Accuracy = ", acc)
    if verbosity >= 2:
        print("Unique labels = ", np.unique(y_train, return_counts=True))
        print("Unique preds = ", np.unique(clf.predict(X_train), return_counts=True))
        print("Rules = ", clf.rl_.rules)
    if verbosity >= 0:
        print("-----------------------------------------------------------------")


    rules_possibilities_list = []
    rules_entropy_list = []

    for j in range(len(clf.rl_.rules)):
        aRule = clf.rl_.rules[j]
        rule_support = sum(aRule['train_labels'])
        if aRule['antecedents'] == [0]: # default decision
            # number of possible worlds is all but those of the prefix's rules
            n_possible_worlds_rule = single_example_possibilities - np.sum(rules_possibilities_list)
        else: # general case
            n_possible_worlds_rule = capt_rl(j, clf.rl_.rules) # call recursive formula
            n_elementary_tokens += len(aRule['antecedents'])
            n_branches_rules += 1
        rule_single_example_entropy = compute_entropy_single_example(n_possible_worlds_rule)

        rules_possibilities_list.append(n_possible_worlds_rule)
        rules_entropy_list.append(rule_single_example_entropy * rule_support)

    if verbosity >= 2:
        print("# possible worlds per rule: ", rules_possibilities_list)
        print("total entropy per rule: ", rules_entropy_list)
    
    reconstructed_dataset_entropy = sum(rules_entropy_list)
    entropy_reduction_ratio = reconstructed_dataset_entropy/no_knowledge_dataset_entropy

elif method == "DL8.5":
    # DL8.5 parameters
    max_depth_param = 2

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

    def explore_tree(tree, splits_list=[]): # Completes leaves_possibilities_list and leaves_entropy_list
        global n_elementary_tokens, leaves_possibilities_list, leaves_entropy_list, leaves_support, n_branches_rules
        if 'error' in tree: # leaf
            n_branches_rules += 1
            leaf_possibilities = num_possibilities(splits_list)
            #if verbosity >= 2:
            #    print("Leaf: %d possibilities." %leaf_possibilities)
            leaves_possibilities_list.append(leaf_possibilities)
            leaf_entropy = compute_entropy_single_example(leaf_possibilities)
            leaves_entropy_list.append(leaf_entropy * tree['support'])
            leaves_support.append(tree['support'])
        else: # internal node (at least one child)
            n_elementary_tokens += 1
            new_splits_list_left = splits_list+ [tree['feat']]
            new_splits_list_right = splits_list+ [tree['feat']]
            if 'left' in tree:
                explore_tree(tree['left'], splits_list=new_splits_list_left)
            if 'right' in tree:
                explore_tree(tree['right'], splits_list=new_splits_list_right)
                
    clf = DL85Classifier(max_depth=max_depth_param, min_sup = trees_min_sample_leaves, fast_error_function=error, time_limit=max_time)
    start = time.perf_counter()
    clf.fit(X_train, y_train)
    duration = time.perf_counter() - start
    if verbosity >= 0:
        print("Model built. Duration of building =", round(duration, 4))
        print("Accuracy DL8.5 on training set =", round(clf.accuracy_, 4))
        
        complete_tree(clf.tree_) # self-made procedure to explicitly compute, check and append the support of each leaf (already contained via #errors and probas but rounded)
        print_tree(clf.tree_)

        print("-----------------------------------------------------------------")

    explore_tree(clf.tree_)

    assert(sum(leaves_support) == n_samples) # better double checking

    if verbosity >= 2:
        print("# possible worlds per rule: ", leaves_possibilities_list)
        print("total entropy per rule: ", leaves_entropy_list)


    reconstructed_dataset_entropy = sum(leaves_entropy_list)
    entropy_reduction_ratio = reconstructed_dataset_entropy/no_knowledge_dataset_entropy
    # print the tree
    import graphviz
    dot = clf.export_graphviz()
    graph = graphviz.Source(dot, format=plot_extension)
    graph.render("./DL8.5_tree_%s" %dataset)

elif method == "sklearn_DT":
    # Sklearn DecisionTreeClassifier parameters
    max_depth_param = 2
    
    clf = DecisionTreeClassifier(random_state=1+random_state_value, max_depth=max_depth_param, min_samples_leaf=trees_min_sample_leaves)

    start = time.perf_counter()
    clf.fit(X_train, y_train)
    duration = time.perf_counter() - start
    train_acc = clf.score(X_train, y_train)
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
    leaves_support = []

    for a_possible_split in nodes_features:
        if a_possible_split != -2:
            n_elementary_tokens += 1
    for a_branch in all_branches:
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
        leaves_support.append(leaf_support)

    assert(sum(leaves_support) == n_samples) # better double checking

    if verbosity >= 2:
        print("# possible worlds per rule: ", leaves_possibilities_list)
        print("total entropy per rule: ", leaves_entropy_list)

    reconstructed_dataset_entropy = sum(leaves_entropy_list)
    entropy_reduction_ratio = reconstructed_dataset_entropy/no_knowledge_dataset_entropy

    from sklearn.tree import plot_tree
    import matplotlib.pyplot as plt
    plot_tree(clf, filled=True)#, feature_names = features)
    plt.savefig("./sklearn_tree_%s.%s" %(dataset, plot_extension), bbox_inches='tight')
    plt.clf()

else:
    raise ValueError("Unknown method " + str(method))

if verbosity >= 0:
    print("Reconstructed dataset joint entropy (with no knowledge) = ", reconstructed_dataset_entropy)
    print("Distance to actual (deterministic) dataset (for %d elementary tokens, and %d branches or rules) = " %(n_elementary_tokens, n_branches_rules), entropy_reduction_ratio)

# Save results in a unified format