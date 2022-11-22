import numpy as np 

def compute_entropy_single_example(n_possibilities):
    one_possibility_proba = (1/n_possibilities) # consider that all possibilities have the same probability
    one_possibility_val = one_possibility_proba * np.log2(one_possibility_proba)
    entropy = - n_possibilities * one_possibility_val
    return entropy
