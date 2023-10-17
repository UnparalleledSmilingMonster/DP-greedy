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
 	

def Q_roots(beta):
    """
    Returns the roots of polynomial Q(Y) = - beta Y**2 + (1-beta)Y - 1  when they exist.
    """
    beta_1 = 3-2*np.sqrt(2)
    beta_2 = 3+2*np.sqrt(2)
    
    if beta > beta_1 and beta < beta_2 : 
        raise Exception("Q has no real roots for these values of beta.")
    
    if beta == beta_1 or beta == beta_2:
        raise Exception("Q admits a single root in that case, please pick another value for beta")
    
    y_1 = (1-beta + np.sqrt((1-beta)**2 -4*beta))/(2*beta)
    y_2 = (1-beta - np.sqrt((1-beta)**2 -4*beta))/(2*beta)
    
    return y_1, y_2
 	
 	
def local_sensitivity_gini(x):
    """
    Returns the local sensitivity for the gini impurity evaluated in x>0, x integer.
    """
    assert(x>0)         #x positive
    assert(int(x)==x)   #x integer
    return 1 - (x/(x+1))**2 - (1/(x+1))**2


def smooth_sensitivity_gini_function(x,beta,t):
    """
    Returns the smooth sensitivity for the gini impurity evaluated in t >0 integer. Xi function in week6.tex
    
    """
    assert(beta>0)
    assert(x>0)         #x positive
    assert(int(x)==x)   #x integer
    assert(t>0)         #x positive
    assert(int(t)==t)   #x integer
        
    return np.exp(-beta*t) * local_sensitivity_gini(max(1,x -t))
    
     

print(exponential(1, 1, [1,2,3] ))

print(laplace(1,1,10))

L = [1,2,3]
print(L[:1])

a,b = [0,1]
print(a,b)
