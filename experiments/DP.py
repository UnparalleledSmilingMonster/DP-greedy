import numpy as np
from scipy import integrate, optimize


beta_1 = 3-2*np.sqrt(2)
beta_2 = 3+2*np.sqrt(2)

def generalized_cauchy_pdf(x,a):
    return 1/(1+np.abs(x)**a)
    

def cauchy_smooth(beta, x, gamma, sensitivity):
    """
    Uses inverse transform sampling method to sample the element from the Cauchy like distribution.
    Efficient to draw 1 sample but costly for lots of samples... The program should be enhanced here because it takes quite some time.
    """
    assert gamma>1
    
    if gamma == 2: #we know a close form solution : arctan'(x) = 1/(1+x^2) and int[-inf, z ] 1/(1+x^2) dx = arctan(z) + pi/2
                   # we add the normalization factor pi so that int[-inf, +inf ] 1/[pi*(1+x^2)] dx  = 1
                   #solve for z : arctan(z)/ pi = u -1/2 ==> z = tan( pi *u - pi/2)
                   # this is well defined because pi *u - pi/2 lies in [-pi/2, pi/2]
    
        u = np.random.uniform(0,1)
        eta = np.tan(np.pi *u - np.pi/2)       
    
    else : 
        valid = False 
        while not valid: #in the very rare cases where the sampled element is too extreme, sample again (very meager bias)
            u = np.random.uniform(0,1)
            #print("Sampled probability:",u)            

            normalization = integrate.quad(lambda x : generalized_cauchy_pdf(x,gamma),0,np.inf)[0]
            #print(integrate.quad(lambda x : generalized_cauchy_pdf(x,gamma),0,np.inf)[0]/normalization) #this should be 0.5
            #print(integrate.quad(lambda x : generalized_cauchy_pdf(x,gamma),0,1e5)[0]/normalization)
            
            equation = lambda z: integrate.quad(lambda x :generalized_cauchy_pdf(x,gamma)/normalization, 0, z)[0] - u
            z_solution = optimize.root_scalar(equation, bracket=[0, 1e5]) #Experimentally 1e5 is the highest value we can go up to before 
                                                                           # the integral computation becomes wrong        

            if z_solution.converged:
                valid = True
                eta = ( 1 if np.random.uniform() < 0.5 else -1) * z_solution.root
            else:
                valid=False
    
   
    #if sensitivity/beta * eta >1 : print(sensitivity/beta * eta )
    
    return sensitivity/beta * eta
    
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
    utility = np.float128(utility)  #to prevent overflows
    probs = np.exp(epsilon* utility/(2*sensitivity) )
    probs/= np.sum(probs)
    
    return np.random.choice(np.arange(0,n), size=1, p=None)
 	

def Q_roots(beta):
    """
    Returns the roots of polynomial Q(Y) = - beta Y**2 + (1-beta)Y - 1  when they exist.
    """
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
    assert x>0, "x should be positive"         
    assert int(x)==x, "x should be an integer"  
    return 1 - (x/(x+1))**2 - (1/(x+1))**2


def smooth_sensitivity_gini_function(x,beta,t,min_supp=1):
    """
    Returns the smooth sensitivity for the gini impurity evaluated in t >0 integer. Xi function in week6.tex    
    """
    #TODO: take into account minimum rule support lambda, add the param
    assert min_supp >0
    assert int(min_supp) == min_supp
    
    assert beta>0
    assert x>0, "x should be positive"         
    assert int(x)==x, "x should be an integer"  
    assert t>=0, "t should be positive: {0}".format(t)  
    assert int(t)==t, "t should be an integer" 
        
    return np.exp(-beta*t) * local_sensitivity_gini(max(min_supp,x -t))
    

def smooth_sensitivity_gini(x, beta, min_supp = 1):
    """
    Returns the smooth sensitivity for a given x>0.
    """
    if beta == beta_1 or beta == beta_2:
        raise Exception("Q admits a single root in that case, please pick another value for beta")
    
    default = smooth_sensitivity_gini_function(x,beta, 0, min_supp)
        
    if beta < beta_1 or beta > beta_2:
        y1,y2 = Q_roots(beta)
        
        t1 = x - y1
        t2 = x - y2
        upper_b = x - min_supp  #>0 
        
        if t2 < x - min_supp :
            xi_t2m = smooth_sensitivity_gini_function(x,beta, np.floor(t2))
            xi_t2p = smooth_sensitivity_gini_function(x,beta, np.ceil(t2))
            if t1<0:            
                return max(xi_t2m, xi_t2p)
            
            else :            
                return max(default, xi_t2m, xi_t2p)   
        
        else :  #t2 >= x - min_supp
            if t1 < 0 :
                return smooth_sensitivity_gini_function(x,beta, upper_b) 
            
            else : 
                return max(default, smooth_sensitivity_gini_function(x,beta, upper_b) )   
                
    
    return default
     
    
"""
print(exponential(1, 1, [1,2,3] ))

print(laplace(1,1,10))

L = [1,2,3]
print(L[:1])

a,b = [0,1]
print(a,b)
"""
