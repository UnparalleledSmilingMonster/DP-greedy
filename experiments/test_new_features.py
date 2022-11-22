from corels import *
import numpy as np 

max_time = 3600 # seconds
max_memory = 8000 # megabytes
n_iter_param = 10 ** 9
maximum_depth = 15
maximum_width = 1
min_support_param = 0.1
policy_param = 'lower_bound'
X, y, features, prediction = load_from_csv("data/compas.csv")
total_features = X.shape[1]
n_samples = y.size

verbosity = 2 # >=2 basic script infos >=3 CORELS infos >= 5 basic info about recursive computations >= 10 detailed info about recursive computations
corels_verbosity = []
n_examples_to_improve_f_obj = 10
cValue = 0.0 #0.9*(1/n_samples) #n_examples_to_improve_f_obj/X.shape[0]

if verbosity >= 3:
    corels_verbosity.extend(['progress', 'mine'])
    

def compute_entropy_single_example(n_possibilities):
    one_possibility_proba = (1/n_possibilities) # consider that all possibilities have the same probability
    one_possibility_val = one_possibility_proba * np.log2(one_possibility_proba)
    entropy = - n_possibilities * one_possibility_val
    return entropy

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

if verbosity >= 2:
    print("Dataset shape: ", X.shape)

    print("-----------------------------------------------------------------")

clf = CorelsClassifier(c=cValue, verbosity=corels_verbosity, policy=policy_param, n_iter=n_iter_param, max_card=maximum_width, max_length=maximum_depth, min_support=min_support_param) #, min_support=0.20)

clf.fit(X, y, features=features, time_limit=max_time, memory_limit=max_memory)

acc = clf.score(X, y)

if verbosity >= 2:
    print("Status = ", clf.get_status())
    print((clf.rl_))
    print("Accuracy = ", acc)
    print("Unique labels = ", np.unique(y, return_counts=True))
    print("Unique preds = ", np.unique(clf.predict(X), return_counts=True))
    print("Rules = ", clf.rl_.rules)

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

rules_possibilities_list = []
rules_entropy_list = []
n_elementary_tokens = 0

for j in range(len(clf.rl_.rules)):
    aRule = clf.rl_.rules[j]
    rule_support = sum(aRule['train_labels'])
    if aRule['antecedents'] == [0]: # default decision
        # number of possible worlds is all but those of the prefix's rules
        n_possible_worlds_rule = single_example_possibilities - np.sum(rules_possibilities_list)
    else: # general case
        n_possible_worlds_rule = capt_rl(j, clf.rl_.rules) # call recursive formula
        n_elementary_tokens += len(aRule['antecedents'])
    rule_single_example_entropy = compute_entropy_single_example(n_possible_worlds_rule)

    rules_possibilities_list.append(n_possible_worlds_rule)
    rules_entropy_list.append(rule_single_example_entropy * rule_support)

if verbosity >= 2:
    print("# possible worlds per rule: ", rules_possibilities_list)
    print("total entropy per rule: ", rules_entropy_list)
    
reconstructed_dataset_entropy = sum(rules_entropy_list)
entropy_reduction_ratio = reconstructed_dataset_entropy/no_knowledge_dataset_entropy

if verbosity >= 2:
    print("Reconstructed dataset joint entropy (with no knowledge) = ", reconstructed_dataset_entropy)
    print("Distance to actual (deterministic) dataset (for %d elementary tokens) = " %n_elementary_tokens, entropy_reduction_ratio)
