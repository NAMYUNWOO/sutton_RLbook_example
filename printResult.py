import pandas as pd
import numpy as np
def printPolicy(k):
    pol = pd.read_csv("policy_%d.csv"%k,header=None,dtype=int)
    pol = np.array(pol)
    for idx in range(len(pol),-1,-1):
        if idx == len(pol):
            print("{0:^3}".format(" "),end=" ")
            print("{0:^3}".format(0),end="")
            for _ in range(len(pol[0])-2):
                print("{0:^3}".format("-"),end="")
            print("{0:^3}".format(len(pol)-1))
            print("\n")
            continue
        i = pol[idx]
        if idx == len(pol)-1:
            print("{0:^3}".format(20),end=" ")
        elif idx == 0:
            print("{0:^3}".format(0),end=" ")
        else:
            print("{0:^3}".format("|"),end=" ")
        for j in i:
            print("{0:^3}".format(j),end="")
        print("\n")

def printValue(k):
    val = pd.read_csv("value_%d.csv"%k,header=None)
    val = val.applymap(lambda x:round(x,1))
    val = np.array(val)
    for idx in range(len(val),-1,-1):
        if idx == len(val):
            print("{0:^6}".format(" "),end=" ")
            print("{0:^6}".format(0),end="")
            for _ in range(len(val[0])-2):
                print("{0:^6}".format("-"),end="")
            print("{0:^6}".format(len(val)-1))
            print("\n")
            continue
        i = val[idx]
        if idx == len(val)-1:
            print("{0:^6}".format(20),end=" ")
        elif idx == 0:
            print("{0:^6}".format(0),end=" ")
        else:
            print("{0:^6}".format("|"),end=" ")
        for j in i:
            print("{0:^6}".format(j),end="")
        print("\n")
