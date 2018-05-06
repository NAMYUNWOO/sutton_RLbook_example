import pandas as pd
import numpy as np
def printPolicy(k):
    pol = pd.read_csv("policy_%d.csv"%k,header=None,dtype=int)
    for i in np.array(pol):
        for j in i:
            print("{0:^3}".format(j),end="")
        print("\n")

def printValue(k):
    val = pd.read_csv("value_%d.csv"%k,header=None)
    val = val.applymap(lambda x:round(x,1))
    for i in np.array(val):
        for j in i:
            print("{0:^6}".format(j),end="")
        print("\n")
