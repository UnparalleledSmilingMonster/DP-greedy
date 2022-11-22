from pydl85 import DL85Classifier
from corels import *
import time
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np 

max_depth_param = 4

X, y, features, prediction = load_from_csv("data/compas.csv")
verbosity = 2 # >=2 basic script infos >=3 CORELS infos >= 5 basic info about recursive computations >= 10 detailed info about recursive computations
    
total_features = X.shape[1]
n_samples = y.size

if verbosity >= 2:
    print("Dataset shape: ", X.shape)

    print("-----------------------------------------------------------------")

def compute_entropy_single_example(n_possibilities):
    one_possibility_proba = (1/n_possibilities) # consider that all possibilities have the same probability
    one_possibility_val = one_possibility_proba * np.log2(one_possibility_proba)
    entropy = - n_possibilities * one_possibility_val
    return entropy

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
        capt = np.ones(shape=(X.shape[0]))
        for a_feat in splits_list:
            if a_feat > 0:
                a_feat -= 1
                capt = np.logical_and(capt, X[:,a_feat])
            elif a_feat < 0:
                a_feat = abs(a_feat)
                a_feat -= 1
                capt = np.logical_and(capt, np.logical_not(X[:,a_feat]))
            else:
                raise ValueError("a_feat = " + str(a_feat) + " < 0")
        leaf_support = np.sum(capt)
        tree["support"] = leaf_support
        # Check if OK with DL8.5 computed error at leaf
        capt_pos = np.sum(np.logical_and(capt, y))
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
    
# return the error and the majority class
def error(sup_iter):
    supports = list(sup_iter)
    maxindex = np.argmax(supports)
    return sum(supports) - supports[maxindex], maxindex

clf = DL85Classifier(max_depth=max_depth_param, fast_error_function=error, time_limit=600)
start = time.perf_counter()
clf.fit(X, y)
duration = time.perf_counter() - start

if verbosity >= 2:
    print("Model built. Duration of building =", round(duration, 4))
    print("Accuracy DL8.5 on training set =", round(clf.accuracy_, 4))
    
    complete_tree(clf.tree_) # self-made procedure to explicitly compute, check and append the support of each leaf (already contained via #errors and probas but rounded)
    print_tree(clf.tree_)

    print("-----------------------------------------------------------------")

single_example_possibilities = num_possibilities([]) #2**X.shape[1] # Binary features

single_example_entropy = compute_entropy_single_example(single_example_possibilities) 
#original_entropies = np.full(y.shape, single_example_entropy)
#no_knowledge_dataset_entropy = np.sum(original_entropies)
no_knowledge_dataset_entropy = n_samples * single_example_entropy
if verbosity >= 2:
    print("Number of possible reconstructions: %d for %d examples." %(single_example_possibilities, n_samples))
    print("Original dataset entropy (with no knowledge) = ", no_knowledge_dataset_entropy)
    print("-----------------------------------------------------------------")




leaves_possibilities_list = []
leaves_entropy_list = []
leaves_support = []
n_elementary_tokens = 0


def explore_tree(tree, splits_list=[]): # Completes leaves_possibilities_list and leaves_entropy_list
    global n_elementary_tokens, leaves_possibilities_list, leaves_entropy_list, leaves_support
    if 'error' in tree: # leaf
        leaf_possibilities = num_possibilities(splits_list)
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

explore_tree(clf.tree_)

assert(sum(leaves_support) == n_samples) # better double checking

if verbosity >= 2:
    print("# possible worlds per rule: ", leaves_possibilities_list)
    print("total entropy per rule: ", leaves_entropy_list)


reconstructed_dataset_entropy = sum(leaves_entropy_list)
entropy_reduction_ratio = reconstructed_dataset_entropy/no_knowledge_dataset_entropy

if verbosity >= 2:
    print("Reconstructed dataset joint entropy (with no knowledge) = ", reconstructed_dataset_entropy)
    print("Distance to actual (deterministic) dataset (for %d elementary tokens) = " %n_elementary_tokens, entropy_reduction_ratio)

# print the tree
import graphviz
dot = clf.export_graphviz()
graph = graphviz.Source(dot, format="png")
graph.render("./test_tree")
