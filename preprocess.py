import pandas as pd
from collections import defaultdict
from FeatExtract import *

def makeBrandPop(data,attrName):
    #制作属性:商品商标被访问次数
    dicts = defaultdict(lambda : 0)
    for line in data[attrName]:
        dicts[line] += 1
    data["brand_pop"] = data["item_brand_id"].apply(lambda x : dicts[x])
    return data

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

# def makePreHit(data):
#    return data[["is_trade","first_cate","user_id","context_timestamp"]].groupby(["is_trade","first_cate","user_id"]).count()


def pre():
    data = pd.read_csv("test.csv",sep=" ")
    # data = pd.read_csv("test.csv")
    data = make_inter(data)
    data.to_csv("te.csv",index= False)

    data = pd.read_csv("train.csv",sep=" ")
    # data = pd.read_csv("test.csv")
    data = make_inter(data)
    data.to_csv("tr.csv",index= False)

# process(data).to_csv("te.csv",index = False)

# train = pd.read_csv("tr1.csv")
# test = pd.read_csv("te1.csv")
# train,test = shop_feat(train,test)
# train.to_csv("tr2.csv",index = False)
# test.to_csv("te2.csv",index = False)
