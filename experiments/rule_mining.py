"""
This file is from the HybridCorels repository of Julien Ferry : 
https://github.com/ferryjul/HybridCORELS/blob/master/paper/rule_mining.py 
and https://github.com/ferryjul/HybridCORELS/blob/master/paper/exp_utils.py 

I'm reusing them with his explicit agreement.
"""

import pandas as pd 
import numpy as np
from scipy.sparse import csc_matrix
from sklearn.ensemble import RandomForestClassifier
from fim import fpgrowth
import itertools
import operator
from scipy.sparse import csc_matrix


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import os

from folktables import ACSDataSource, ACSIncome, ACSEmployment, generate_categories
import DP as dp
from corels import load_from_csv, RuleList, CorelsClassifier




def generate_rulespace(X, y, max_card=2, min_support=1, neg_columns=False, method='fpgrowth', random_state=42):
    """
    Generate the set of all rules for a given task

    Arguments
    ---------
    X : pd.DataFrame, shape = [n_samples, n_features]
        The training input samples. All features must be binary.
    y : array-like,  shape = [n_samples,]
        The binary target for the task
    max_card : int
        The maximal cardinality of the rules
    min_support : int
        Minimum coverage of the rules (in percent e.g. 1, 5, etc.)
    neg_columns : Boolean
        Include the negation of X columns as well
    method : str
        Mining method is either 'rf' of 'fpgrowth'

    Returns
    -------
    X : pd.DataFrame, shape = [n_samples, 2*n_features]
        The training input samples new features representing negation of existing columns.
    rules : List( List(str) ), lenght = n_rules
        Rules are represented as a list of strings
    coverage : List( float ), length = n_rules
        Coverage of each rule in [0, 1]
    """
    
    # Rule Mining
    if method == 'fpgrowth': # Generate rules with fpgrowth

        # FPGrowth treats each instance as a set of items [[1 2 3], [3 4], [4 6], ...]
        itemMatrix = [[item for item in X.columns if row[item]==1] for _, row in X.iterrows()] 
        tuple_rules = fpgrowth(itemMatrix, supp=min_support, zmin=1, zmax=max_card)
        coverage = [float(rule[1] / X.shape[0]) for rule in tuple_rules]
        rules = [list(rule[0]) for rule in tuple_rules]
    
        if neg_columns:
            X_neg = 1 - X
            neg_coverage = list(X_neg.mean(0))
            neg_rules = ['not_' + name.strip() for name in X.columns]
            X_neg.columns = neg_rules

            # Update rules
            rules = rules + [[r] for r in neg_rules]
            coverage = coverage + neg_coverage
            X = pd.concat([X, X_neg], axis=1)

        
    else: # Using Random Forests to mine rules

        # Rules associated with positive preditions
        prules = []
        for length in range(2, max_card+1, 1):
            n_estimators = 250 * length
            clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=length, random_state=random_state)
            clf.fit(X, y)
            for n in range(n_estimators):
                prules.extend(extract_rules_rf(clf.estimators_[n], X.columns))

        # Rules associated with negative predictions
        nrules = []
        for length in range(2, max_card, 1):
            n_estimators = 250 * length
            clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=length, random_state=random_state)
            clf.fit(X, 1-y)
            for n in range(n_estimators):
                nrules.extend(extract_rules_rf(clf.estimators_[n], X.columns))

        # Add negation after fitting the rf
        X_neg = 1 - X
        X_neg.columns = ['not_' + name.strip() for name in X.columns]
        X = pd.concat([X, X_neg], axis=1)


    return X, rules, coverage




def screen_rules(rules, X, coverage, N_rules):
    """
    Filter rules based on minimum support and return
    new binary data.

    Arguments
    ---------
    rules : List( List(str) ), lenght = n_rules
        Rules are represented as a list of strings
    X : pd.DataFrame, shape = [n_samples, n_features]
        The training input samples. All features must be binary.
        The rules must be subsets of columns of X.
    N_rules : int
        Maximum number of rules to consider

    Returns
    -------
    RMatrix : np.array, shape = [n_samples, N_rules]
        Binary matrix indicating if instance i satisfies rule j
    rules : List(str)
        list of all considered rules
    """
    
    # Map column_name to column index
    itemInd = {}
    for i, name in enumerate(X.columns):
        itemInd[name] = int(i)
    
    len_rules = [len(rule) for rule in rules]
    indices = np.array(list(itertools.chain.from_iterable([[itemInd[x] for x in rule] for rule in rules])))
    indptr = list(accumulate(len_rules))
    indptr.insert(0, 0) # insert 0 at 0 position/necessary for building csc-matrix
    indptr = np.array(indptr)
    data = np.ones(len(indices))
    ruleMatrix = csc_matrix((data,indices,indptr), shape=(len(X.columns), len(rules)))
    
    mat = np.matrix(X) * ruleMatrix # a matrix of data sum wrt rules
    lenMatrix = np.matrix([len_rules for _ in range(X.shape[0])])
    # Z_ij == 1 if instance i respects rule j
    Z = (mat == lenMatrix).astype(int)
    C = np.array(np.sum(Z, axis=0))[0]
    assert np.isclose(C/X.shape[0], coverage).all()

    # Only keep the N_rules rules with largest coverage
    ind = np.argsort(coverage)[::-1][:N_rules].tolist()
    rules = [rules[i] for i in ind]
    
    RMatrix = np.array(Z[:, ind]) 
    
    return RMatrix, rules



def extract_rules_rf(tree, feature_names):
    left     = tree.tree_.children_left
    right    = tree.tree_.children_right
    features = [feature_names[i] for i in tree.tree_.feature]
    # Get ids of leaf nodes
    idx = np.argwhere(left == -1)[:, 0]

    def recurse(left, right, child, lineage=None):          
        if lineage is None:
            lineage = []
        if child in left:
            parent = np.where(left == child)[0].item()
            suffix = 'not_'
        else:
            parent = np.where(right == child)[0].item()
            suffix = ''

        lineage.append((suffix + features[parent].strip()))

        if parent == 0:
            lineage.reverse()
            return lineage
        else:
            return recurse(left, right, parent, lineage)
    
    rules = []
    for child in idx:
        rule = []
        for node in recurse(left, right, child):
            rule.append(node)
        rules.append(rule)
    return rules



def accumulate(iterable, func=operator.add):
    # accumulate([1,2,3,4,5]) --> 1 3 6 10 15
    # accumulate([1,2,3,4,5], operator.mul) --> 1 2 6 24 120
    it = iter(iterable)
    total = next(it)
    yield total
    for element in it:
        total = func(total, element)
        yield total



def mine_rules_preprocessing(X, y, max_card, min_support, n_rules):
    
    X, rules, coverage = generate_rulespace(X, y, max_card=max_card, min_support=min_support, 
                                            neg_columns=True, method="fpgrowth")
    RMatrix, rules = screen_rules(rules, X, coverage, N_rules=n_rules)

    # Reformat rules
    rules = [" && ".join(rule) for rule in rules]
    df = pd.DataFrame(RMatrix, columns=rules)
    return df
    

def get_data(dataset, max_card=2, min_support=1, n_rules=300, random_state_param=42):
    # # Generate the acs_data if it is not already there
    # if dataset == "acs_employ":# and not os.path.exists(f"data/{dataset}.csv"):
    #     generate_acs_data()

    # Mine the dataset set if it has not already been done

    df = pd.read_csv(f"data/{dataset}_unfiltered.csv", sep = ',')
    X = df.iloc[:, :-1]
    prediction = df.iloc[:, -1].name
    y = np.array(df.iloc[:, -1])

    # Mine the rules
    X = mine_rules_preprocessing(X, y, max_card, min_support, n_rules)
    features = list(X.columns)
    X = np.array(X)

    # Save the dataset
    df = pd.DataFrame(np.column_stack((X, y)), columns=features+[prediction])
    for column in df:
        new_name = column
        new_name= column.replace('''"''',"")
        new_name=new_name.replace(",","")
        df=df.rename(columns={column: new_name})
    df.to_csv(f"data/{dataset}.csv", encoding='utf-8', index=False)




def to_df(X, features):
    df_X = {}
    for key, val in X.items():
        df_X[key] = pd.DataFrame(val, columns=features)
    return df_X


    
    
    
if __name__ == "__main__": 
 
    get_data("folktable")
        
