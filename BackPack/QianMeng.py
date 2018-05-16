import time
import pandas as pd
from utils import *
import lightgbm as lgb
from sklearn.metrics import log_loss
from collections import defaultdict
import sklearn

def prop_score(trdata,tedata,propThreshold):
    pdata = trdata[["is_trade","item_property_list"]]
    tradeDict = defaultdict(lambda :0)
    freqDict = defaultdict(lambda : 0)
    for line in pdata.values:
        prop_list = line[1].split(";")
        for i in prop_list:
            freqDict[i]+=1
            if(int(line[0]) == 1):
                tradeDict[i] += 1
    trscore = []
    for line in pdata["item_property_list"]:
        trscore.append(prop_tool(tradeDict,freqDict,line,propThreshold))
    tescore = []
    for line in tedata["item_property_list"]:
        tescore.append(prop_tool(tradeDict,freqDict,line,propThreshold))

    trdata["prop_score"] = trscore
    tedata["prop_score"] = tescore

    trdata.loc[:,"prop_score"] = pd.Series(trscore)
    tedata.loc[:,"prop_score"] = pd.Series(tescore)
    return trdata,tedata

def prop_tool(tradeDict,freqDict,props,threshold):
    prop_list = props.split(";")
    sum = 0
    for i in prop_list:
        if(freqDict[i] > threshold):
            sum += (tradeDict[i] * 1.0 / freqDict[i])
    return sum

def timestamp_datetime(value):
    format = '%Y-%m-%d %H:%M:%S'
    value = time.localtime(value)
    dt = time.strftime(format, value)
    return dt


def convert_data(data):
    data['time'] = data.context_timestamp.apply(timestamp_datetime)
    data['day'] = data.time.apply(lambda x: int(x[8:10]))
    data['hour'] = data.time.apply(lambda x: int(x[11:13]))
    user_query_day = data.groupby(['user_id', 'day']).size(
    ).reset_index().rename(columns={0: 'user_query_day'})
    data = pd.merge(data, user_query_day, 'left', on=['user_id', 'day'])
    user_query_day_hour = data.groupby(['user_id', 'day', 'hour']).size().reset_index().rename(
        columns={0: 'user_query_day_hour'})
    data = pd.merge(data, user_query_day_hour, 'left',
                    on=['user_id', 'day', 'hour'])

    #print(data.groupby(["is_trade","user_query_day_hour"]).size())
    return data

def hasTrade(trdata,tedata):
    pdata = trdata[["is_trade","first_cate","user_id"]].drop_duplicates(inplace=False)
    pdata = (pdata.loc[pdata["is_trade"]==1])[["first_cate","user_id"]]
    pdata["has_trade"] = 1
    pdata = pdata[["first_cate","user_id","has_trade"]]
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

if __name__ == "__main__":
    online = True# 这里用来标记是 线下验证 还是 在线提交

def processHasTrade(test,labelName):
    test.loc[test["has_trade"]==1,[labelName]] *= 1.1
    test.loc[test["has_trade"]==0,[labelName]] *= 0.9

if __name__ == "__main__":
    pass
    # online = False# 这里用来标记是 线下验证 还是 在线提交
    # data = pd.read_csv('tr.csv')
    # data.drop_duplicates(inplace=True)
    # data = convert_data(data)
    # if online == False:
    #     train = data.loc[data.day < 24]  # 18,19,20,21,22,23,24
    #     test = data.loc[data.day == 24]  # 暂时先使用第24天作为验证集
    # elif online == True:
    #     train = data.copy()
    #     test = pd.read_csv('te.csv')
    #     test = convert_data(test)
    # train,test=cateHit(train,test,"second_cate")
    # train,test = hasTrade(train,test)
    #
    # #train,test = prop_score(train,test,10000)
    # features = ['cate_hit','item_id','item_brand_id', 'item_city_id', 'item_price_level', 'item_sales_level',
    # train,test = prop_score(train,test,1000)
    # train,test=cateHit(train,test,"second_cate")
    # train,test = hasTrade(train,test)
    #
    # train,test = prop_score(train,test,10000)
    # features = ['prop_score','cate_hit','item_id','item_brand_id', 'item_city_id', 'item_price_level', 'item_sales_level',
    #
    #             'item_collected_level', 'item_pv_level', 'user_gender_id','user_occupation_id',
    #             'user_age_level', 'user_star_level', 'user_query_day', 'user_query_day_hour',
    #             'context_page_id', 'hour', 'shop_review_num_level', 'shop_star_level',
    #             'shop_review_positive_rate', 'shop_score_service','shop_score_description',
    #             'cate_inter','prop_inter','second_cate', #"has_trade"
    #             ]
    # target = ['is_trade']
    #
    # if online == False:
    #     clf = lgb.LGBMClassifier(learning_rate=0.1,num_leaves=63, max_depth=7, n_estimators=80, n_jobs=20)
    #     clf.fit(train[features], train[target], feature_name=features,
    #             categorical_feature=['user_gender_id'])
    #     test['lgb_predict'] = clf.predict_proba(test[features])[:, 1]
    #     processHasTrade(test,"lgb_predict")
    #     print(log_loss(test[target], test['lgb_predict']))
    # else:
    #     clf = lgb.LGBMClassifier(num_leaves=63, max_depth=7, n_estimators=80, n_jobs=20)
    #     clf.fit(train[features], train[target],
    #             categorical_feature=['user_gender_id'])
    #     test['predicted_score'] = clf.predict_proba(test[features])[:, 1]
    #     processHasTrade(test,"predicted_score")
    #     test[['instance_id', 'predicted_score']].to_csv('hithas.csv', index=False,sep=' ')#保存在线提交结果
