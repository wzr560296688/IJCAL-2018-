import pandas as pd
import time
from collections import defaultdict

def hasTradeStat(test):
    testA = test.loc[test["has_trade"]==1]
    ratio = len(testA.loc[testA["is_trade"] == 1].values) * 1.0 / len(testA.values)
    testB = test.loc[test["has_trade"]==0]
    ratio2 = len(testB.loc[testB["is_trade"] == 1].values) * 1.0 / len(testB.values)
    print("has {0} hasnot {1}".format(ratio,ratio2))


def hasTrade(trdata,tedata,cateName):
    pdata = trdata[["is_trade",cateName,"user_id"]].drop_duplicates(inplace=False)
    pdata = (pdata.loc[pdata["is_trade"]==1])[[cateName,"user_id"]]
    pdata["has_trade"] = 1
    pdata = pdata[[cateName,"user_id","has_trade"]]
    tedata = pd.merge(tedata, pdata, 'left')
    tedata.loc[tedata["has_trade"]!=1,"has_trade"]=0
    trdata = pd.merge(trdata, pdata, 'left'   )
    trdata.loc[trdata["has_trade"]!=1,"has_trade"]=0
    return trdata,tedata

def processHasTrade(test,labelName):
    test.loc[test["has_trade"]==1,[labelName]] *= 1.1
    test.loc[test["has_trade"]==0,[labelName]] *= 0.9

def cateHit(trdata,tedata,cateName):
    pdata = trdata[[cateName,"user_id"]].groupby([cateName,"user_id"]).size().reset_index().rename(columns={0:"cate_hit"})
    tedata = pd.merge(tedata, pdata, 'left')
    tedata.fillna(0)
    #tedata.loc[tedata["cate_hit"]!=1,"cate_hit"]=0
    trdata = pd.merge(trdata, pdata, 'left')
    trdata.fillna(0)
    #trdata.loc[trdata["cate_hit"]!=1,"cate_hit"]=0
    return trdata,tedata

def make_inter(data):
    feature = ['item_category_list','item_property_list','predict_category_property', 'context_timestamp']
    unuseddata = data[feature]
    cate_inter = []
    prop_inter = []
    ipcp_pl_len = []
    for indexs in unuseddata.index:
        items = unuseddata.loc[indexs].values[0:]
        icl = items[0]
        ipl = items[1]
        ipcp = items[2]
        icl_set = set(icl.split(";"))
        ipl_set = set(ipl.split(";"))
        ipcp_cl = list(ipcp.split(";"))
        ipcp_cl_set = set()
        ipcp_pl_set = set()
        for i in range(len(ipcp_cl)):
            c_p = ipcp_cl[i].split(":")
            ipcp_cl_set.add(c_p[0])
            if len(c_p) < 2 or c_p[1] == -1:
                continue
            ipcp_pl_set = ipcp_pl_set.union(set(c_p[1].split("--")))
        cate_inter.append(len(icl_set.intersection(ipcp_cl_set)))
        prop_inter.append(len(ipl_set.intersection(ipcp_pl_set)))
        ipcp_pl_len.append(len(ipcp_pl_set))
    data['cate_inter'] = pd.Series(cate_inter,index=data.index)
    data['prop_inter'] = pd.Series(prop_inter,index=data.index)

    first_cate = []
    second_cate = []
    for line in data["item_category_list"]:
        first_cate.append(line.split(";")[0])
        if(len (line.split(";")) < 2):
            second_cate.append("{0}".format(-1))
        else:
            second_cate.append(line.split(";")[1])
    data["first_cate"] = first_cate
    data["second_cate"] = second_cate
    return data

#def overSample()

#No need, because brand_id has beed added as a feature of lightGBM
# def makeBrandPop(data,attrName):
#     #制作属性:商品商标被访问次数
#     dicts = defaultdict(lambda : 0)
#     for line in data[attrName]:
#         dicts[line] += 1
#     data["brand_pop"] = data["item_brand_id"].apply(lambda x : dicts[x])
#     return data

def makePreHit(data):
   return data[["is_trade","first_cate","user_id","context_timestamp"]].groupby(["is_trade","first_cate","user_id"]).count()

# def timestamp_datetime(value):
#     format = '%Y-%m-%d %H:%M:%S'
#     value = time.localtime(value)
#     dt = time.strftime(format, value)
#     return dt


# def convert_data(data):
#     data['time'] = data.context_timestamp.apply(timestamp_datetime)
#     data['day'] = data.time.apply(lambda x: int(x[8:10]))
#     data['hour'] = data.time.apply(lambda x: int(x[11:13]))
#     user_query_day = data.groupby(['user_id', 'day']).size(
#     ).reset_index().rename(columns={0: 'user_query_day'})
#     data = pd.merge(data, user_query_day, 'left', on=['user_id', 'day'])
#     user_query_day_hour = data.groupby(['user_id', 'day', 'hour']).size().reset_index().rename(
#         columns={0: 'user_query_day_hour'})
#     data = pd.merge(data, user_query_day_hour, 'left',
#                     on=['user_id', 'day', 'hour'])
#     # data.loc[data["context_page_id"] ==1,'context_page_id'] = 0
#     # data.loc[data["context_page_id"] >1,'context_page_id'] = 1
#
#     #print(data.groupby(["is_trade","user_query_day_hour"]).size())
#     return data
