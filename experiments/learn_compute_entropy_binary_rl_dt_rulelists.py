from corels import *
from HeuristicRL import GreedyRLClassifier
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
parser.add_argument('--expe_id', type=int, default=0, choices= [0,1,2,3,4,5], help='method-dataset combination (for now, only COMPAS and tic-tac-toe supported)')
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
max_memory = 11000 # megabytes

# Slurm task parallelism
expe_id=args.expe_id
datasets = ["compas", "tic-tac-toe", "adult"]
methods = ["CORELS", "GreedyRL"] # 0 for CORELS, 1 for DL8.5, 2 for sklearn DT (CART)    
slurm_expes = []
for d in datasets:
    for m in methods:
        slurm_expes.append([d, m])

dataset = slurm_expes[expe_id][0]
method = slurm_expes[expe_id][1]

if verbosity >= 0:
    print("Slurm #expes = ", len(slurm_expes))
    print("Current expe: dataset %s, method %s" %(dataset, method))

# MPI parallelism
random_seeds = [i for i in range(5)] # for 1) data train/test split and 2) methods initialization
min_support_params = [0.01*i for i in range(1,6)] # minimum proportion of training examples that a rule (or a leaf) must capture
max_depth_params = [i for i in range(1,11)]
max_width_params = [1,2,3] #[i for i in range(1,4)]

configs_list = []
for rs in random_seeds: # 5 values
    for msp in min_support_params: # 5 values
        for mdp in max_depth_params: # 10 values
            for mwp in max_width_params: # 2 values
                configs_list.append([rs, msp, mdp, mwp])

random_state_value = configs_list[rank][0]
min_support = configs_list[rank][1]
max_depth = configs_list[rank][2]
max_width = configs_list[rank][3]

if verbosity >= 0:
    print("MPI #params = ", len(configs_list))
    print("Current expe: random_state_value %d, min_support %.3f, max_depth %d, max_width %d" %(random_state_value, min_support, max_depth, max_width))

# Load the data
if dataset in ["compas", 'adult']:
    X, y, features, prediction = load_from_csv("data/%s.csv" %dataset)
else: # datasets from the pydl8.5 repository
    data_full = np.genfromtxt("data/%s.txt" %dataset, delimiter=' ')
    X = data_full[:, 1:]
    y = data_full[:, 0]
    X = X.astype('int32')
    y = y.astype('int32')
    features, prediction = ["feature_%d" %i for i in range(X.shape[1])], "pred"

if dataset == "adult": # need to subsample
    from sklearn.utils.random import sample_without_replacement
    selected = sample_without_replacement(y.size, int(0.1*y.size), random_state=random_state_value)
    X = X[selected,:]
    y = y[selected]
    
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_ratio, random_state=random_state_value)
total_features = X_train.shape[1]
n_samples = y_train.size
if verbosity >= 0:
    print("Dataset shape: ", X.shape)
    print("Training set shape: ", X_train.shape)
    print("-----------------------------------------------------------------")

X = "shouldnotbeusedanymore"
y = "shouldnotbeusedanymore"

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
leaves_single_example_entropy_list = []
leaves_support = []

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

if method == "CORELS":
    # CORELS-specific parameters
    n_iter_param = 10 ** 9
    policy_param = 'objective'
    # greater than zero to help pruning
    # but smaller than 1/n_samples because we don't want to trade-off accuracy
    cValue =  0.99*(1/n_samples) #n_examples_to_improve_f_obj/X.shape[0]

    clf = CorelsClassifier(c=cValue, verbosity=corels_verbosity, policy=policy_param, n_iter=n_iter_param, max_card=max_width, max_length=max_depth, min_support=min_support) #, min_support=0.20)

elif method == "GreedyRL":

    clf = GreedyRLClassifier(min_support=min_support, max_length=max_depth, verbosity=corels_verbosity, max_card=max_width, allow_negations=True)

else:
    print("method " + method + "unknown.")
    exit()

start = time.perf_counter()
clf.fit(X_train, y_train, features=features, time_limit=max_time, memory_limit=max_memory, prediction_name=prediction)
duration = time.perf_counter() - start

train_acc = np.mean(y_train == clf.predict(X_train))
test_acc = np.mean(y_test == clf.predict(X_test))

if verbosity >= 0:
    print("Model built. Duration of building =", round(duration, 4))
    print("Status = ", clf.get_status())
    print((clf.rl_))
    print("Accuracy = ", train_acc)
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
    leaves_single_example_entropy_list.append(rule_single_example_entropy)
    leaves_support.append(rule_support)
    average_tokens_per_examples += (rule_support * n_elementary_tokens)
    n_elementary_tokens_path += n_elementary_tokens
if verbosity >= 2:
    print("# possible worlds per rule: ", rules_possibilities_list)
    print("total entropy per rule: ", rules_entropy_list)
reconstructed_dataset_entropy = sum(rules_entropy_list)
entropy_reduction_ratio = reconstructed_dataset_entropy/no_knowledge_dataset_entropy

assert(sum(leaves_support) == n_samples) # better double checking
average_tokens_per_examples /= n_samples
## average_tokens_per_examples, n_elementary_tokens_path, leaves_support, leaves_single_example_entropy_list
model = str(clf.rl_)
res = [[dataset, method, random_state_value, min_support, max_depth, max_width, duration, train_acc, test_acc, reconstructed_dataset_entropy, no_knowledge_dataset_entropy, n_elementary_tokens, n_branches_rules, entropy_reduction_ratio, average_tokens_per_examples, n_elementary_tokens_path, leaves_support, leaves_single_example_entropy_list, model, clf.get_status()]]

# Gather the results for the 5 folds on process 0
if ccanada_expes:
    res = comm.gather(res, root=0)

if rank == 0 or not ccanada_expes:
    # save results
    fileName = './results/%s_%s.csv' %(method, dataset) #_expes_optimal_vs_heuristic
    import csv
    with open(fileName, mode='w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(['dataset', 'method', 'random_state_value', 'min_support', 'max_depth', 'max_width', 'duration', 'train_acc', 'test_acc', 'reconstructed_dataset_entropy', 'no_knowledge_dataset_entropy', 'n_elementary_tokens', 'n_branches_rules', 'entropy_reduction_ratio', 'average_tokens_per_examples', 'n_elementary_tokens_path', 'leaves_support', 'sorted_leaves_single_example_entropy_list', 'model', 'status'])
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