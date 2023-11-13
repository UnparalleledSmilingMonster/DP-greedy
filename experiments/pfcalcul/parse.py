import os
import ast

"""
dic_format = {'dataset':0, 'max_length':1, 'Mechanism':2, epsilon_letter:3, delta_letter:4, lambda_letter:5, "Confidence":6, 'C_max':7, 'N':8, 'Runs':9, 'Avg. Time(s)':10, 'Accuracy':11}
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
    epsilons.remove('x')
    
    for i in range(len(L)):
        L[i].append(True if L[i][1] == "vanilla" else False)
        L[i].append(True if L[i][1] == "vanilla" else False)
        L[i].append(False)
                
    for dataset in datasets:
        buff = [elt[-4] if (elt[0] == dataset and elt[2]!="vanilla") else -1 for elt in L]
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

def GreedyRLParser(directory):
    res = {}
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        # checking if it is a file
        with open(f, 'r') as file :
            line = file.readline() #we only read the first line as the rest is warnings
            data = ast.literal_eval(line)
            key = "".join(str(data[0:8])) #primary key (not accounting for the seed = repetition)
            
            if key in res :
                res[key][9] += 1
                res[key][10] += float(data[-2])  #time 
                res[key][11] += float(data[-1])  #accuracy
            else :
                res[key] = data[0:9]
                if res[key][2].startswith("smooth"): res[key][2] = res[key][2].replace("smooth", "sm")
                elif res[key][2].startswith("global"):res[key][2] = res[key][2].replace("global", "gl")
                res[key].append(1)
                res[key].append(float(data[-2]))  #time 
                res[key].append(float(data[-1]))    
        
    for key in res :
        res[key][10] = float(pformat(res[key][10]/res[key][9], num=2))
        res[key][11] = float(pformat(res[key][11]/res[key][9]))     
                    
    return res
                

        

if __name__ == '__main__':    
    epsilon_letter='\u03B5'
    delta_letter='\u03B4'
    lambda_letter = '\u03BB'
    params = ['dataset', 'Mechanism', epsilon_letter, delta_letter, lambda_letter, "Confidence", 'C_max', 'N', 'Runs', 'Avg. Time(s)', 'Accuracy']

    res = GreedyRLParser("results")
    latex_tabular("tex/results.tex", params, res)
  

