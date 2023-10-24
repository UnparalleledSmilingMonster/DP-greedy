import DP as dp
import numpy as np
import matplotlib.pyplot as plt
import seaborn
seaborn.set_theme()

rng = np.random.default_rng(2002)

N = 1000
x = np.linspace(20,5000, N).astype(int)
epsilon = 1
gamma = 2
avg = 1

print("This small code aims at comparing Laplace noise for global sensitivity VS Cauchy noise for smooth sensitivity") 
print("We fix a DP budget epsilon =", epsilon)


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
