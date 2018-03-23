import pandas as pd
from collections import defaultdict
def makeBrandPop(data,attrName,threshold):
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

data = pd.read_csv("tr1.csv")
print(data["first_cate"])
#data = make_inter(data)
#data = makeBrandPop(data,"item_brand_id",1000)
#data.loc[(data["brand_pop"]<1000,"item_brand_id")] = -1
#print(len(data["item_brand_id"].drop_duplicates(inplace=False).values))
#data.to_csv("tr1.csv",index = False)
# data = pd.read_csv("test.csv")
# process(data).to_csv("te.csv",index = False)
