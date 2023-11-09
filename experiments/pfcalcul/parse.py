#! /usr/bin/env python

from rocknrun import *


# Parsers should return a dict stat-name -> value-list. value-list is a list of the different values that the stat takes during a run
class GreedyRLParser(object):

    def __call__(self,output):
        res = {}
        for line in output:
            data = [dat.split(' ') for dat in line.rstrip().split('--')]
            for d in data :
                if len(d) == 2 :
                    s = d[0].strip()
                    val = d[1].strip()
                    if not s in res :
                        res[s] = [int(val)]
                    else :
                        res[s].append(int(val))
                else :
                    res['outcome'] = d
        return res
                

        

if __name__ == '__main__':    
    e = Experiment()
    parsers = dict([(m, SatireParser()) for m in e.all_methods])
    o = Observation(e, parsers)
    o.write_summary_table('tex/summary.tex', ['number of conflicts', 'cpu time'], precisions=[[0],[2]])

