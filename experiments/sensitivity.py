import DP as dp
import numpy as np
import matplotlib.pyplot as plt
import seaborn
seaborn.set_theme()

rng = np.random.default_rng(2002)

N = 1000
M = 5000
x = np.linspace(20,M, N).astype(int)
epsilon = 1
delta = 1 /M**2

avg = 200

def cauchy_VS_laplace():
    print("This small code aims at comparing Laplace noise for global sensitivity VS Cauchy noise for smooth sensitivity") 
    print("We fix a DP budget epsilon =", epsilon)

    gamma = 2
    beta = epsilon/(2*(gamma+1))
    min_supps = [10,50,100,500,1000]

    plt.figure(figsize=(12,8))
    plt.title("Comparison of Smooth Sensivity and Global Sensitivity for Gini Impurity", fontsize = 18)
    plt.xlabel("Number of Samples caught", fontsize = 18)
    plt.ylabel("Noise distortion imposed", fontsize = 18)
    plt.figtext(0.5, 0, "$\epsilon = {0} \ | \ Runs = {1}$".format(epsilon, avg), ha="center", fontsize=18)


    for min_supp in min_supps:
        noise_cauchy = np.zeros(N)
        min_idx = 0
        for i in range(N):
            if x[i] < min_supp : 
                min_idx +=1
                continue
            sensi_cauchy = dp.smooth_sensitivity_gini(x[i],beta, min_supp = min_supp) 
            for j in range(avg):             
                noise_cauchy[i] += dp.cauchy_smooth(beta, x[i], gamma, sensi_cauchy)
        
        noise_cauchy/=avg
        plt.plot(x[min_idx:], noise_cauchy[min_idx:], label="Cauchy Noise "+ r'$\Lambda ={0}$'.format(min_supp), linewidth = 2)
        print("Cauchy Noise : min_supp ={0} : ".format(min_supp) + "Var = {0}".format(np.var(noise_cauchy)))
            

    noise_lap = np.zeros(N)
    for i in range(N):
        for j in range(avg):
            sensi_lap = 0.5
            noise_lap[i] += dp.laplace(epsilon, sensi_lap, 1)[0]
    noise_lap/= avg
    plt.plot(x, noise_lap, label="Laplace Noise")
    print("Laplace Noise : Var = {0}".format(np.var(noise_lap)))
    plt.legend()
    plt.show()
    
    
def laplace_glb_vs_smooth():
    print("This small code aims at comparing Laplace noise for global sensitivity VS smooth sensitivity") 
    print("We fix a DP budget epsilon={0}, delta={1}(smooth Laplace)".format(epsilon,delta))

    gamma = 2
    beta = epsilon/(2*np.log(2/delta))
    #min_supps = [10,100,1000]
    min_supps = [100]
    
    plt.figure(figsize=(12,8))
    plt.title("Comparison of Smooth Sensivity and Global Sensitivity for Gini Impurity", fontsize = 18)
    plt.xlabel("Number of Samples caught", fontsize = 18)
    plt.ylabel("Noise distortion imposed", fontsize = 18)
    plt.figtext(0.5, 0, "$\epsilon = {0} \ | \ \delta = {1}$".format(epsilon, delta), ha="center", fontsize=18)            

    noise_lap = np.zeros((N,avg))
    for i in range(N):
        sensi_lap = 0.5
        for j in range(avg):
            noise_lap[i,j] = np.abs(dp.laplace(epsilon, sensi_lap, 1)[0])
    noise_lap_avg = np.average(noise_lap, axis=1)
    noise_lap_var = np.var(noise_lap, axis=1)
    plt.plot(x, noise_lap_avg , label="Laplace Global Noise")
    plt.fill_between(x, noise_lap_avg-noise_lap_var, noise_lap_avg+noise_lap_var, alpha = 0.3)

    print("Laplace Global Noise : Var = {0}".format(np.var(noise_lap)))
    
    for min_supp in min_supps:
        noise_lap_smooth = np.zeros((N,avg))
        min_idx = 0
        for i in range(N):
            if x[i] < min_supp : 
                min_idx +=1
                continue
            sensi_lap_smooth = dp.smooth_sensitivity_gini(x[i],beta, min_supp = min_supp) 
            for j in range(avg):             
                noise_lap_smooth[i,j] = np.abs(dp.laplace_smooth(epsilon, sensi_lap_smooth))
        
        noise_lap_smooth_avg = np.average(noise_lap_smooth, axis=1)
        noise_lap_smooth_var = np.var(noise_lap_smooth, axis=1)
        
        noise_lap_smooth/=avg
        plt.plot(x[min_idx:], noise_lap_smooth_avg[min_idx:], label="Laplace Smooth Noise "+ r'$\Lambda ={0}$'.format(min_supp), linewidth = 2)
        plt.fill_between(x[min_idx:], noise_lap_smooth_avg[min_idx:]-noise_lap_smooth_var[min_idx:], \
         noise_lap_smooth_avg[min_idx:]+noise_lap_smooth_var[min_idx:], alpha = 0.3)
        print("Laplace Smooth Noise : min_supp ={0} : ".format(min_supp) + "Var = {0}".format(np.var(noise_lap_smooth)))
        
        
    plt.yscale("log")
    plt.legend()
    plt.show()
    

laplace_glb_vs_smooth()
    
    
