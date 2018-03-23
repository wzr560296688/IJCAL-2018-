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
def timestamp_datetime(value):
    format = '%Y-%m-%d %H:%M:%S'
    value = time.localtime(value)
    dt = time.strftime(format, value)
    return dt
data= pd.read_csv("test.csv")
# user_query_day = data.groupby(['user_id', 'day']).size(
#     ).reset_index().rename(columns={0: 'user_query_day'})
data['time'] = data.context_timestamp.apply(timestamp_datetime)
data['day'] = data.time.apply(lambda x: int(x[8:10]))
data['hour'] = data.time.apply(lambda x: int(x[11:13]))
user_query_day = data.groupby(['user_id', 'day']).size().reset_index()
print(user_query_day)
