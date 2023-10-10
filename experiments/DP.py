import numpy as np
  
    
def laplace(epsilon, sensitivity, n): 
    """
    Implementation of the Laplace mechanism.
    Returns a noise vector following a Laplace distribution : Lap(sensitivity/epsilon) of dim n.
    """
    assert sensitivity >=0
    assert epsilon > 0 	
    return np.random.laplace(scale = sensitivity/epsilon, size = n)


def exponential(epsilon, sensitivity, utility):
    """
    Implementation of the Exponential Mechanism.
    Return the index of the element sampled through exponential distribution according to the utility array.
    """
    n =len(utility)
    if not isinstance(utility, np.ndarray) : utility = np.array(utility)
    probs = np.exp(epsilon/(2*sensitivity) * utility)
    probs/= np.sum(probs)
    
    return np.random.choice(np.arange(0,n), size=1, p=None)
 	
 	
 	
print(exponential(1, 1, [1,2,3] ))

print(laplace(1,1,10))

L = [1,2,3]
print(L[:1])

a,b = [0,1]
print(a,b)
