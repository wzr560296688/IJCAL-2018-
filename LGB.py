import time
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import log_loss
from collections import defaultdict

# def filterFreq(data,attrName,threshold):
#     dicts = defaultdict(lambda : 0)
#     for line in data[attrName]:
#         dicts[line] += 1
#     for i in range(len(data.values)):
#         if(dicts[data.loc[i,attrName]] < threshold):
#             data.loc[i,attrName] = -1
#     # print(data[attrName])
#     #data.loc[(data[attrName] not in dicts < threshold,attrName)] = -1123

data= pd.read_csv("tr.csv")
print(len(data.values))
