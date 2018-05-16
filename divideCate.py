import pandas as pd
from collections import defaultdict
import time
freq = defaultdict(lambda  :0)
lb = 0.3
ub = 0.6
propertyName = "item_property_list"
def timestamp_datetime(value):
    format = '%Y-%m-%d %H:%M:%S'
    value = time.localtime(value)
    dt = time.strftime(format, value)
    return dt
def divdeByCate(data,attrName):
    cates = data.loc[:,[attrName]].drop_duplicates(subset = attrName)
    for i in cates[attrName].values:
        makeFreq(data[data[attrName]==i],"item_property_list")
        data.loc[data[attrName]==i,[propertyName]] = data.loc[data[attrName]==i,[propertyName]].apply(filterProp)
        break
def makeFreq(oneCate,attrName):
     global freq
     freq = defaultdict(lambda  :0)
     for line in oneCate[attrName].values:
        ls = line.split(";")
        for i in ls:
             freq[i] += 1
     for i in freq.keys():
        freq[i] = freq[i] * 1.0 / len(oneCate.index)
     for i in list(freq.keys()):
        if(freq[i] <  lb or freq[i] >  ub):
            freq.pop(i)
def filterProp(oneTuple):
    ls = str(oneTuple).split(";")
    tmp = ""
    for i in ls:
        if i in freq.keys():
            tmp += (i + ";")
    return oneTuple
# data = pd.read_csv("tr.csv")
# data['realtime'] = data['context_timestamp'].apply(timestamp_datetime)
# data['realtime'] = pd.to_datetime(data['realtime'])
# data['day'] = data['realtime'].dt.day
# data['hour'] = data['realtime'].dt.hour
# #print(data)
# divdeByCate(data,"second_cate")
