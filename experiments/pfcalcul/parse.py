import os
import ast

def pformat(var, mode ="f", num=3 ):
    if (var is None or var==0) : return 'x'
    form = "{" + "0:.{0}{1}".format(num,mode) + "}"
    return form.format(var)
         
    
def order(dic):
    datasets = set([dic[key][0] for key in dic])
    L=list(dict(sorted(dic.items())).values())
    for i in range(len(L)):
        L[i].append(True if L[i][1] == "vanilla" else False)
                
    for dataset in datasets:
        buff = [elt[-2] if (elt[0] == dataset and elt[1]!="vanilla") else -1 for elt in L]
        print(buff)
        idx = buff.index(max(buff))
        L[idx][-1] = True    
    
    return L
    
    
def latex_tabular(filename, params, dic):
    L = order(dic)
    m = len(params)
    columns = "".join(["|c" for _ in range(m)])+"|"
    tabular = """\\begin{figure}[h!]\n\\begin{center}\n\\begin{tabular}{""" + columns + """}\n\\hline\n"""
    
    fst_line = ""
    for i in range(m):
        fst_line += params[i] + "&"        
    tabular += fst_line[:-1] + "\\\\\n \\hline \\hline\n"
    
    line = ""
    n = len(L[0])
    for result in L :
        line = ""
        for i in range(n-1):
            line +=  "\\textbf{" + str(result[i]) +"}"+ "&" if result[n-1] else str(result[i]) + "&" #in bold if best result
        tabular += line[:-1] + "\\\\\n \\hline\n"    
    
    tabular += """\\end{tabular}\n\\end{center}\n\\caption{Comparing DP methods and Vanilla Greedy RL}\n\\end{figure}"""
    
    with open(filename, 'w') as f :
        f.write(tabular)
        f.close()

def GreedyRLParser(directory):
    res = {}
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        # checking if it is a file
        with open(f, 'r') as file :
            for line in file:
                data = ast.literal_eval(line)
                key = "".join(str(data[0:6])) #primary key (not accounting for the seed = repetition)
                if key in res :
                    res[key][8] += 1
                    res[key][9] += float(data[-2])  #time 
                    res[key][10] += float(data[-1])  #accuracy
                else :
                    res[key] = data[0:8]
                    res[key].append(1)
                    res[key].append(float(data[-2]))  #time 
                    res[key].append(float(data[-1]))    
        
    for key in res :
        res[key][9] = float(pformat(res[key][9]/res[key][8]))
        res[key][10] = float(pformat(res[key][10]/res[key][8]))     
                    
    return res
                

        

if __name__ == '__main__':    
    epsilon_letter='\u03B5'
    delta_letter='\u03B4'
    lambda_letter = '\u03BB'
    params = ['dataset', 'Mechanism', epsilon_letter, delta_letter, lambda_letter, "Confidence", 'C_max', 'N', 'Runs', 'Avg. Time(s)', 'Accuracy']

    res = GreedyRLParser("results")
    latex_tabular("tex/results.tex", params, res)
    """
    e = Experiment()
    parsers = dict([(m, GreedyRLParser()) for m in e.all_methods])
    o = Observation(e, parsers)
    o.write_summary_table('tex/results.tex', )
    """
