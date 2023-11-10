#! /usr/bin/env python

from rocknrun import *

def pformat(var, mode ="f", num=3 ):
    if (var is None or var==0) : return 'x'
    form = "{" + "0:.{0}{1}".format(num,mode) + "}"
    return form.format(var)
    

# Parsers should return a dict stat-name -> value-list. value-list is a list of the different values that the stat takes during a run
class GreedyRLParser(object):

    def __call__(self,output):
        res = {}
        for line in output:
            data = [line.rstrip().split(' ')]
            key = "".join(data[0:6]) #primary key (not accounting for the seed = repetition)
            if key in res :
                res[s][7] += 1
                res[s][8] += data[-2]  #time 
                res[s][9] += data[-1]  #accuracy
            else :
                res[s][0:6] = data[0:6]
                res[s][7] = 1
                res[s][8] = data[-2]  #time 
                res[s][9] = data[-1]      
        
        for key in res :
            res[key][8] = pformat(res[key][8]/res[key][7])
            res[key][9] = pformat(res[key][9]/res[key][7])            
                
        return res
                

        

if __name__ == '__main__':    
    epsilon_letter='\u03B5'
    delta_letter='\u03B4'
    lambda_letter = '\u03BB'
    e = Experiment()
    parsers = dict([(m, GreedyRLParser()) for m in e.all_methods])
    o = Observation(e, parsers)
    o.write_summary_table('tex/summary.tex', ['dataset', 'Mechanism', epsilon_letter, delta_letter, lambda_letter, "Confidence", 'C_max', 'N', 'Runs', 'Avg. Time(s)', 'Accuracy'])

