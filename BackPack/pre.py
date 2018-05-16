import pandas as pd
from collections import defaultdict
from utils import *


# data = pd.read_csv("train.csv")
#makePreHit(data).to_csv("haha.csv")
data = pd.read_csv("train.csv")
data = make_inter(data)
data.to_csv("tr.csv",index= False)


data = pd.read_csv("test.csv")
data = make_inter(data)
data.to_csv("te.csv",index= False)
#data = makeBrandPop(data,"item_brand_id")
# data.loc[(data["brand_pop"]<200,"item_brand_id")] = -1
#print(len(data["item_brand_id"].drop_duplicates(inplace=False).values))
#data.to_csv("tr1.csv",index = False)
# data = pd.read_csv("test.csv")
#data.to_csv("tr.csv",index= False)

# process(data).to_csv("te.csv",index = False)
