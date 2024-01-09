import os
import ast
import json 
import numpy as np

"""
dic_format = {'dataset':0, 'max_length':1, 'Mechanism':2, epsilon_letter:3, delta_letter:4, lambda_letter:5, 'N':6, 'Runs':7, 'Avg. Time(s)':8, 'Accuracy_train':9, 'Accuracy_test':10, 'var_acc_train':11, 'var_acc_test':12}}
"""

def pformat(var, mode ="f", num=3 ):
    if (var is None or var==0) : return 'x'
    form = "{" + "0:.{0}{1}".format(num,mode) + "}"
    return form.format(var)

def highlight(L, dic):
    """
    Adds values to rows to indicate that the row should be highlighted in the result tables. The criteria are : dataset, max_length and epsilon
    e.g: Best results for given max_length (and given dataset)
    """

    datasets = set([dic[key][0] for key in dic])
    max_lengths = set([dic[key][1] for key in dic])
    epsilons = set([dic[key][3] for key in dic])
    #epsilons.remove('x')
    
    for i in range(len(L)):
        L[i].append(True if L[i][2] == "vanilla" else False)
        L[i].append(True if L[i][2] == "vanilla" else False)
        L[i].append(False)
                
    for dataset in datasets:
        buff = [elt[-6] if (elt[0] == dataset and elt[2]!="vanilla") else -1 for elt in L]
        idx = buff.index(max(buff))
        L[idx][-3] = True    
        for max_length in max_lengths:
            buff2 = [buff[i] if (L[i][1] == max_length and L[i][2]!="vanilla") else -1 for i in range(len(L))]
            idx = buff2.index(max(buff2))
            L[idx][-2] = True           
            for epsilon in epsilons:
                buff3 = [buff2[i] if (L[i][3] == epsilon) else -1 for i in range(len(L))]
                idx = buff3.index(max(buff3))
                L[idx][-1] = True  
    return L         
    
def order(dic):
    L=list(dict(sorted(dic.items())).values())    
    L = highlight(L, dic)
    
    return L
    
    
def latex_tabular(filename, params, dic):
    L = order(dic)
    m = len(params)
    columns = "".join(["|c" for _ in range(m)])+"|"
    tabular = """\\begin{longtable}{""" + columns + """}\n\\caption{Comparing DP methods and Vanilla Greedy RL}\\\\\n\\hline\n"""
    
    fst_line = ""
    for i in range(m):
        fst_line += params[i] + "&"        
    tabular += fst_line[:-1] + "\\\\\n \\hline \\hline\n"
    
    emph = ["\\textcolor{{blue}}{{{0}}}", "\\textbf{{{0}}}", "\\hl{{{0}}}"]
    line = ""
    n = len(L[0])
    for result in L :
        line = ""
        for i in range(n-3):
            form = str(result[i])
            for j in range(3):
                form = emph[j].format(form) if result[-(j+1)] else form    #in bold if best result            
            line +=  form + "&" 
        tabular += line[:-1] + "\\\\\n \\hline\n"    
    
    tabular += """\\end{longtable}\n"""
    
    with open(filename, 'w') as f :
        f.write(tabular)
        f.close()

def GreedyRLParser(directory="results", summary = "summary.nfo"):
    res = {}
    
    if not os.path.exists(summary):
        for filename in os.listdir(directory):
            f = os.path.join(directory, filename)
            # checking if it is a file
            with open(f, 'r') as file :
                line = file.readline() #we only read the first line as the rest is warnings
                #print(line)
                data = ast.literal_eval(line)
                key = "".join(str(data[0:7])) #primary key (not accounting for the seed = repetition)
                
                if key in res :
                    res[key][7] += 1
                    res[key][8] += float(data[-3])  #time 
                    res[key][9] += float(data[-2])  #accuracy
                    res[key][10] += float(data[-1])  #accuracy
                    res[key][11].append(float(data[-2]))
                    res[key][12].append(float(data[-1]))
                else :
                    res[key] = data[0:7]
                    if res[key][2].startswith("smooth"): res[key][2] = res[key][2].replace("smooth", "sm")
                    elif res[key][2].startswith("global"):res[key][2] = res[key][2].replace("global", "gl")
                    res[key].append(1)
                    res[key].append(float(data[-3]))  #time 
                    res[key].append(float(data[-2]))  #train acc 
                    res[key].append(float(data[-1]))  #test acc
                    res[key].append([float(data[-2])])
                    res[key].append([float(data[-1])])  
   
                    
        for key in res :
            if res[key][2] != "vanilla":
                res[key][8] = float(pformat(res[key][8]/res[key][7], num=2))  
                res[key][9] = float(pformat(res[key][9]/res[key][7]))
                res[key][10] = float(pformat(res[key][10]/res[key][7]))  
                
                #Computing variances:
                res[key][11] = float(pformat(np.var(res[key][11]), num =5))
                res[key][12] =  float(pformat(np.var(res[key][12]), num =5))
            else : 
                res[key][11] = 'x'
                res[key][12] = 'x'
                    
        with open(summary, "w") as summ:
             summ.write(json.dumps(res)) 
             summ.close()
        return res
     
    else: 
        with open(summary, 'r') as summ :
            res = json.load(summ)
            summ.close()
        return res
        

if __name__ == '__main__':    
    epsilon_letter='\u03B5'
    delta_letter='\u03B4'
    lambda_letter = '\u03BB'
    params = ['dataset', 'Mechanism', epsilon_letter, delta_letter, lambda_letter, 'N', 'Runs', 'Avg. Time(s)', 'Train_acc', 'Test_acc']

    res = GreedyRLParser("visu", "summary_visu.nfo")
    #latex_tabular("tex/results.tex", params, res)
  

