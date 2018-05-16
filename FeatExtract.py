import pandas as pd
import numpy as np
from smooth import *
#商户特征
def shop_feat(tra,test):
    # 从训练集一中获取店铺相关的特征
    store = tra
    # 该店铺被点击次数
    t = store[['shop_id']]
    t['shop_click_total'] = 1
    t = t.groupby('shop_id').agg('sum').reset_index()
    store = pd.merge(store,t,on='shop_id',how='left')
    test = pd.merge(test,t,on='shop_id',how='left')
    ta = t['shop_click_total']
    # 该店铺被购买次数
    t = store[['shop_id','is_trade']]
    t = t.groupby('shop_id').agg('sum').reset_index()
    t.rename(columns={'is_trade':'shop_click_buy_total'},inplace=True)
    tb = t['shop_click_buy_total']
    store = pd.merge(store,t,on='shop_id',how='left')
    test = pd.merge(test,t,on='shop_id',how='left')
    # 该店铺购买率
    # store['shop_click_buy_rate'] = t['shop_click_buy_total']/store['shop_click_total']
    # test['shop_click_buy_rate'] = t['shop_click_buy_total']/test['shop_click_total']
    hyper = HyperParam(1, 1)
    alpha,beta =  hyper.update_from_data_by_FPI(ta,tb, 1000, 0.00000001)
    store['shop_click_buy_rate'] = (t['shop_click_buy_total'] + alpha)/(store['shop_click_total'] + alpha + beta)
    test['shop_click_buy_rate'] = (t['shop_click_buy_total'] + alpha)/(test['shop_click_total'] + alpha + beta)
    # 'shop_difuser_total',
    return store,test

#上下文特征
def context_feat(tra,test):
    # 展示在该页的被点击的次数
    t = tra[['context_page_id']]
    t['page_click']=1
    t = t.groupby('context_page_id').agg('sum').reset_index()
    ta = t['page_click']
    tra = pd.merge(tra,t,on='context_page_id',how='left')
    test = pd.merge(test,t,on='context_page_id',how='left')
    # 展示在该页被购买的次数
    t = tra[['context_page_id','is_trade']]
    t = t.groupby('context_page_id').agg('sum').reset_index()
    t = t.rename(columns={'is_trade':'page_buy'})
    tb = t['page_buy']
    tra = pd.merge(tra,t,on='context_page_id',how='left')
    test = pd.merge(test,t,on='context_page_id',how='left')
    # 展示在该页的购买率
    # tra['page_buy_rate'] = t['page_buy']/tra['page_click']
    # test['page_buy_rate'] = t['page_buy']/test['page_click']
    hyper = HyperParam(1, 1)
    alpha,beta =  hyper.update_from_data_by_FPI(ta,tb, 1000, 0.00000001)
    tra['page_buy_rate'] = (t['page_buy'] + alpha)/(tra['page_click'] + alpha + beta)
    test['page_buy_rate'] = (t['page_buy'] + alpha)/(test['page_click'] + alpha + beta)
    return tra,test

#用户特征
def user_feat(tra,test):
    #提取训练集一的用户相关特征
    # 用户点击次数
    t = tra[['user_id']]
    t['user_click_total'] = 1
    t = t.groupby('user_id')['user_click_total'].agg('sum').reset_index()
    ta = t['user_click_total']
    tra = pd.merge(tra,t,on='user_id',how='left')
    test = pd.merge(test,t,on='user_id',how='left')

    # 用户点击且购买次数
    t = tra[['user_id','is_trade']]
    t = t.groupby('user_id')['is_trade'].agg('sum').reset_index()
    t = t.rename(columns={'is_trade':'user_click_buy_total'})
    tb = t['user_click_buy_total']
    tra = pd.merge(tra,t,on='user_id',how='left')
    test = pd.merge(test,t,on='user_id',how='left')

    # # 用户购买占点击比重
    # tra['user_click_buy_rate'] = t['user_click_buy_total']/tra['user_click_total']
    # test['user_click_buy_rate'] = t['user_click_buy_total']/test['user_click_total']
    hyper = HyperParam(1, 1)
    alpha,beta =  hyper.update_from_data_by_FPI(ta,tb, 1000, 0.00000001)
    tra['user_click_buy_rate'] = (t['user_click_buy_total'] + alpha)/(tra['user_click_total'] + alpha + beta)
    test['user_click_buy_rate'] = (t['user_click_buy_total'] + alpha)/(test['user_click_total'] + alpha + beta)
    return tra,test
    # [['user_click_total','user_click_buy_total','user_click_buy_rate','user_click_difshop_total','user_click_shop_rate','user_click_max','user_click_mean','user_click_min',
    # 'user_id','gosa_click','gosa_buy','gosa_rate','user_gender_id','user_occupation_id','user_star_level','user_age_level'
    #      ]]

# 提取广告商品相关特征
def item_feat(tra,test):
    # 该展示商品被点击次数
    t = tra[['item_id']]
    t['item_click']=1
    t = t.groupby('item_id').agg('sum').reset_index()
    ta = t["item_click"]
    tra = pd.merge(tra,t,on='item_id',how='left')
    test = pd.merge(test,t,on='item_id',how='left')
    # 该展示商品被购买次数
    t = tra[['item_id','is_trade']]
    t = t.groupby('item_id').agg('sum').reset_index()
    t = t.rename(columns={'is_trade':'item_buy'})
    tb = t["item_buy"]
    tra = pd.merge(tra,t,on='item_id',how='left')
    test = pd.merge(test,t,on='item_id',how='left')
    # 该展示商品被购买率

    tra['item_buy_rate'] = t['item_buy']/tra['item_click']
    test['item_buy_rate'] = t['item_buy']/test['item_click']
    hyper = HyperParam(1, 1)
    alpha,beta =  hyper.update_from_data_by_FPI(ta,tb, 1000, 0.00000001)
    tra['item_buy_rate'] = (t['item_buy'] + alpha)/(tra['item_click'] + alpha + beta)
    test['item_buy_rate'] = (t['item_buy'] + alpha)/(test['item_click'] + alpha + beta)

    # 该商品是否属于高销量类别
    t = tra[['second_cate','is_trade']]
    t = tra.groupby('second_cate').agg('sum').reset_index()
    t['is_high_sale'] = t['is_trade'].apply(lambda x:1 if x>=1000 else 0)
    tra = pd.merge(tra,t[['second_cate','is_high_sale']],on='second_cate',how='left')
    test = pd.merge(test,t[['second_cate','is_high_sale']],on='second_cate',how='left')
    return tra,test

#商品属性列表的处理-popular 属性---存疑
def item_prop_feat(tra,test):
    #商品的属性列表处理
    t = tra[['item_id','item_property_list','is_trade']]
    #将属性列表从字符串转变为list
    def getPropertySet(s):
         PropertySet=set(s.split(';'))
         return PropertySet
    t['item_property_list']=t.item_property_list.apply(getPropertySet)
    #获取受欢迎商品的属性
    t1=t[t.is_trade==1]
    Popular=set()
    for i in t1['item_property_list']:
         Popular=set(list(Popular)+list(i))
    #取出每种商品
    t2=t[['item_id','item_property_list']]
    t2.drop_duplicates(['item_id'],inplace=True)
    #记录每个商品的属性总数
    def setSum_propertys(s):
          return len(s)
    t2['Sum_Propertys']=t2.item_property_list.apply(setSum_propertys)
    #记录每个商品的受欢迎属性个数
    t2['Sum_Popular_Propertys']=0
    # t['Sum_Popular_Propertys']=len([i for i in t.item_property_list if i in list(Popular)])
    def getSum_Popular_Propertys(s):
         return len(Popular.intersection(s))
    t2['Sum_Popular_Propertys']=t2.item_property_list.apply(getSum_Popular_Propertys)
    #记录每个商品受欢迎属性占总属性个数百分比
    t2['Popular_rate']=t2['Sum_Popular_Propertys']/t2['Sum_Propertys']
    #合并
    tra=pd.merge(tra,t2[['item_id','Sum_Propertys','Sum_Popular_Propertys','Popular_rate']],on='item_id',how='left')
    test=pd.merge(test,t2[['item_id','Sum_Propertys','Sum_Popular_Propertys','Popular_rate']],on='item_id',how='left')
    return tra,test

#组合特征：品牌-商品
def brand_combine(train,test):
    # 点击该商品品牌次数
    t = train[['item_id','item_brand_id']]
    t['item_brand_click'] = 1
    t = t.groupby(['item_id','item_brand_id']).agg('sum').reset_index()
    train = pd.merge(train,t,on=['item_id','item_brand_id'],how='left')
    test = pd.merge(test,t,on=['item_id','item_brand_id'],how='left')
    # 购买该shang品牌次数
    t = train[['item_id','item_brand_id','is_trade']]
    t = t.groupby(['item_id','item_brand_id']).agg('sum').reset_index()
    t = t.rename(columns={'is_trade':'item_brand_buy'})
    train = pd.merge(train,t,on=['item_id','item_brand_id'],how='left')
    test = pd.merge(test,t,on=['item_id','item_brand_id'],how='left')
    # 购买该shangpin品牌次数占该pinpai总购买比率
    t = train[['item_brand_id','is_trade']]
    t = t.groupby('item_brand_id').agg('sum').reset_index()
    t = t.rename(columns={'is_trade':'brand_buy_total'})
    train = pd.merge(train,t,on='item_brand_id',how='left')
    test = pd.merge(test,t,on=['item_brand_id','item_brand_id'],how='left')
    train['item_brand_rate'] = train['item_brand_buy']/train['brand_buy_total']
    test['item_brand_rate'] = test['item_brand_buy']/test['brand_buy_total']
    return train,test

#组合特征：商品-店铺
def shop_combine(train,test):
    # 从训练集一中获取商品店铺相关的特征
    commodity_shop = train
    # 该商品在该店铺点击次数
    t = commodity_shop[['item_id','shop_id']]
    t['item_shop_click_total'] = 1
    t = t.groupby(['item_id','shop_id']).agg(sum).reset_index()
    commodity_shop = pd.merge(commodity_shop,t,on=['item_id','shop_id'],how='left')
    test = pd.merge(test,t,on=['item_id','shop_id'],how='left')
    # 该商品在该店铺购买次数
    t = commodity_shop[['item_id','shop_id','is_trade']]
    t = t.groupby(['item_id','shop_id']).agg(sum).reset_index()
    t.rename(columns={'is_trade':'item_shop_click_buy_total'},inplace=True)
    commodity_shop = pd.merge(commodity_shop,t,on=['item_id','shop_id'],how='left')
    test = pd.merge(test,t,on=['item_id','shop_id'],how='left')
    # 该商品在该店铺购买率
    commodity_shop['item_shop_click_buy_rate'] = commodity_shop['item_shop_click_buy_total']/commodity_shop['item_shop_click_total']
    test['item_shop_click_buy_rate'] = test['item_shop_click_buy_total']/test['item_shop_click_total']
    # 该商品在该店铺售卖数量/该商品总售卖量
    commodity_shop['item_shop_sale_rate'] = commodity_shop['item_shop_click_buy_total']/commodity_shop['item_buy']
    test['item_shop_sale_rate'] = test['item_shop_click_buy_total']/test['item_buy']
    # 该商品在该店铺售卖数量/该商户总售卖量
    commodity_shop['shop_item_sale_rate'] = commodity_shop['item_shop_click_buy_total']/commodity_shop['shop_click_buy_total']
    test['shop_item_sale_rate'] = test['item_shop_click_buy_total']/test['shop_click_buy_total']
    return commodity_shop,test
    return commodity_shop,test

#组合特征：用户-商品的等级
def user_item_level_combine(train,test):

    #用户点击该价格等级的商品数量
    t = train[['user_id','item_price_level']]
    t['user_price_ctotal'] = 1
    t = t.groupby(['user_id','item_price_level']).agg('sum').reset_index()
    train = pd.merge(train,t,on=['user_id','item_price_level'],how='left')
    test = pd.merge(test,t,on=['user_id','item_price_level'],how='left')
    # 用户购买该价格等级商品的数量
    t = train[['user_id','item_price_level','is_trade']]
    t = t.groupby(['user_id','item_price_level']).agg('sum').reset_index()
    t = t.rename(columns={'is_trade':'user_price_btotal'})
    train = pd.merge(train,t,on=['user_id','item_price_level'],how='left')
    test = pd.merge(test,t,on=['user_id','item_price_level'],how='left')
    # 用户点击/购买该价格等级商品占用户购买比
    train['user_price_crate'] = train['user_price_ctotal']/train['user_click_total']
    test['user_price_crate'] = test['user_price_ctotal']/test['user_click_total']
    train['user_price_brate'] = train['user_price_btotal']/train['user_click_buy_total']
    test['user_price_brate'] = test['user_price_btotal']/test['user_click_buy_total']

    #用户点击该收藏等级的商品数量
    t = train[['user_id','item_collected_level']]
    t['user_collected_ctotal'] = 1
    t = t.groupby(['user_id','item_collected_level']).agg('sum').reset_index()
    train = pd.merge(train,t,on=['user_id','item_collected_level'],how='left')
    test = pd.merge(test,t,on=['user_id','item_collected_level'],how='left')
    # 用户购买该收藏等级商品的数量
    t = train[['user_id','item_collected_level','is_trade']]
    t = t.groupby(['user_id','item_collected_level']).agg('sum').reset_index()
    t = t.rename(columns={'is_trade':'user_collected_btotal'})
    train = pd.merge(train,t,on=['user_id','item_collected_level'],how='left')
    test = pd.merge(test,t,on=['user_id','item_collected_level'],how='left')
    # 用户购买该价格等级商品占用户购买比
    # print(train[['user_collected_ctotal']])
    train['user_collected_crate'] = train['user_collected_ctotal']/train['user_click_total']
    test['user_collected_crate'] = test['user_collected_ctotal']/test['user_click_total']
    train['user_collected_brate'] = train['user_collected_btotal']/train['user_click_buy_total']
    test['user_collected_brate'] = test['user_collected_btotal']/test['user_click_buy_total']
    # 'user_collected_btotal'
    #用户点击该展示等级的商品数量
    train = pd.read_csv('data/train1_f.csv')
    u = pd.read_csv('data/user_feature1.csv')
    t = train[['user_id','item_pv_level']]
    t['user_pv_ctotal'] = 1
    t = t.groupby(['user_id','item_pv_level']).agg('sum').reset_index()
    train = pd.merge(train,t,on=['user_id','item_pv_level'],how='left')
    test = pd.merge(test,t,on=['user_id','item_pv_level'],how='left')
    # 用户购买该价格等级商品的数量
    t = train[['user_id','item_pv_level','is_trade']]
    t = t.groupby(['user_id','item_pv_level']).agg('sum').reset_index()
    t = t.rename(columns={'is_trade':'user_pv_btotal'})
    train = pd.merge(train,t,on=['user_id','item_pv_level'],how='left')
    test = pd.merge(test,t,on=['user_id','item_pv_level'],how='left')
    # 用户购买该价格等级商品占用户购买比
    u = u[['user_id','user_click_total','user_click_buy_total']]
    u = u.drop_duplicates(subset='user_id')
    train = pd.merge(train,u,on=['user_id'],how='left')
    # print(train[['user_collected_ctotal']])
    train['user_pv_crate'] = train['user_pv_ctotal']/train['user_click_total']
    test['user_pv_crate'] = test['user_pv_ctotal']/test['user_click_total']
    train['user_pv_brate'] = train['user_pv_btotal']/train['user_click_buy_total']
    test['user_pv_brate'] = test['user_pv_btotal']/test['user_click_buy_total']

    # 该年龄星级职业
    t = train[['item_price_level','user_occupation_id','user_star_level','user_age_level']]
    t['occupation_star_age_price_total'] = 1
    t = t.groupby(['user_occupation_id','user_star_level','user_age_level','item_price_level']).agg('sum').reset_index()
    train = pd.merge(train,t,on=['user_occupation_id','user_star_level','user_age_level','item_price_level'],how='left')
    test = pd.merge(test,t,on=['user_occupation_id','user_star_level','user_age_level','item_price_level'],how='left')
    # 该职业星级购买该商品数
    t = train[['item_price_level','user_occupation_id','user_star_level','user_age_level','is_trade']]
    t = t.groupby(['user_occupation_id','user_star_level','user_age_level','item_price_level']).agg('sum').reset_index()
    t = t.rename(columns={'is_trade':'occupation_star_age_price_buy_total'})
    train = pd.merge(train,t,on=['user_occupation_id','user_star_level','user_age_level','item_price_level'],how='left')
    test = pd.merge(test,t,on=['user_occupation_id','user_star_level','user_age_level','item_price_level'],how='left')
    # 该职业星级购买该商品率
    train['occupation_star_age_price_buy_rate'] = train['occupation_star_age_price_buy_total']/train['occupation_star_age_price_total']
    test['occupation_star_age_price_buy_rate'] = test['occupation_star_age_price_buy_total']/test['occupation_star_age_price_total']

    # 该年龄点击该类目次数
    t = train[['user_star_level','item_price_level']]
    t['star_price_click'] = 1
    t = t.groupby(['user_star_level','item_price_level']).agg('sum').reset_index()
    train = pd.merge(train,t,on=['user_star_level','item_price_level'],how='left')
    test = pd.merge(test,t,on=['user_star_level','item_price_level'],how='left')
    # 该年龄购买该类目次数
    t = train[['user_star_level','item_price_level','is_trade']]
    t = t.groupby(['user_star_level','item_price_level']).agg('sum').reset_index()
    t = t.rename(columns={'is_trade':'star_price_buy'})
    train = pd.merge(train,t,on=['user_star_level','item_price_level'],how='left')
    test = pd.merge(test,t,on=['user_star_level','item_price_level'],how='left')
    # 该年龄购买该类别率
    train['star_price_rate'] = train['star_price_buy']/train['star_price_click']
    test['star_price_rate'] = test['star_price_buy']/test['star_price_click']

    return train,test

#组合特征：用户-类别
def user_cate_combine(train,test):
    # 该用户点击该类目次数
    t = train[['user_id','second_cate']]
    t['user_cate_click'] = 1
    t = t.groupby(['user_id','second_cate']).agg('sum').reset_index()
    train = pd.merge(train,t,on=['user_id','second_cate'],how='left')
    # 该用户购买该类目次数
    t = train[['user_id','second_cate','is_trade']]
    t = t.groupby(['user_id','second_cate']).agg('sum').reset_index()
    t = t.rename(columns={'is_trade':'user_cate_buy'})
    train = pd.merge(train,t,on=['user_id','second_cate'],how='left')
    # 该用户购买该类别率
    train['user_cate_rate'] = train['user_cate_buy']/train['user_cate_click']
    # # 该用户点击该类目占该用户点击比
    # train = pd.merge(train,u[['user_id','user_click_total','user_click_buy_total']],on='user_id',how='left')
    # train['user_cate_rate'] = train['user_cate_click']/train['user_click_total']
    # # 该用户购买该类目占该用户购买比
    # train['user_cate_brate'] = train['user_cate_buy']/train['user_click_buy_total']
    # # 该年龄
    # # 该年龄点击该类目次数
    # t = train[['user_age_level','cate2']]
    # t['age_cate_click'] = 1
    # t = t.groupby(['user_age_level','cate2']).agg('sum').reset_index()
    # train = pd.merge(train,t,on=['user_age_level','cate2'],how='left')
    # # 该年龄购买该类目次数
    # t = train[['user_age_level','cate2','is_trade']]
    # t = t.groupby(['user_age_level','cate2']).agg('sum').reset_index()
    # t = t.rename(columns={'is_trade':'age_cate_buy'})
    # train = pd.merge(train,t,on=['user_age_level','cate2'],how='left')
    # # 该年龄购买该类别率
    # train['age_cate_rate'] = train['age_cate_buy']/train['age_cate_click']
    # # # 该年龄点击该类目占该年龄点击比
    # #
    # # train['user_cate_rate'] = train['user_cate_click']/train['user_click_total']
    # # # 该年龄购买该类目占该年龄购买比
    # # train['user_cate_brate'] = train['user_cate_buy']/train['user_click_buy_total']
    # train[['age_cate_rate','user_age_level','cate2']].to_csv('data/age_cate_feature1.csv',index=None)
    # # 'age_cate_click','age_cate_buy',
    # # 该星级
    # # 该年龄点击该类目次数
    # t = train[['user_star_level','cate2']]
    # t['star_cate_click'] = 1
    # t = t.groupby(['user_star_level','cate2']).agg('sum').reset_index()
    # train = pd.merge(train,t,on=['user_star_level','cate2'],how='left')
    # # 该年龄购买该类目次数
    # t = train[['user_star_level','cate2','is_trade']]
    # t = t.groupby(['user_star_level','cate2']).agg('sum').reset_index()
    # t = t.rename(columns={'is_trade':'star_cate_buy'})
    # train = pd.merge(train,t,on=['user_star_level','cate2'],how='left')
    # # 该年龄购买该类别率
    # train['star_cate_rate'] = train['star_cate_buy']/train['star_cate_click']
    # # # 该年龄点击该类目占该年龄点击比
    # #
    # # train['user_cate_rate'] = train['user_cate_click']/train['user_click_total']
    # # # 该年龄购买该类目占该年龄购买比
    # # train['user_cate_brate'] = train['user_cate_buy']/train['user_click_buy_total']
    # train[['star_cate_rate','user_star_level','cate2']].to_csv('data/star_cate_feature1.csv',index=None)
    # # 'star_cate_click','star_cate_buy',
    # # 该职业
    # # 该年龄点击该类目次数
    # t = train[['user_occupation_id','cate2']]
    # t['occupation_cate_click'] = 1
    # t = t.groupby(['user_occupation_id','cate2']).agg('sum').reset_index()
    # train = pd.merge(train,t,on=['user_occupation_id','cate2'],how='left')
    # # 该年龄购买该类目次数
    # t = train[['user_occupation_id','cate2','is_trade']]
    # t = t.groupby(['user_occupation_id','cate2']).agg('sum').reset_index()
    # t = t.rename(columns={'is_trade':'occupation_cate_buy'})
    # train = pd.merge(train,t,on=['user_occupation_id','cate2'],how='left')
    # # 该年龄购买该类别率
    # train['occupation_cate_rate'] = train['occupation_cate_buy']/train['occupation_cate_click']
    # # # 该年龄点击该类目占该年龄点击比
    # #
    # # train['user_cate_rate'] = train['user_cate_click']/train['user_click_total']
    # # # 该年龄购买该类目占该年龄购买比
    # # train['user_cate_brate'] = train['user_cate_buy']/train['user_click_buy_total']
    # train[['occupation_cate_rate','user_occupation_id','cate2']].to_csv('data/occupation_cate_feature1.csv',index=None)
    # # 'occupation_cate_click','occupation_cate_buy',
    # # 该类
    # # 该年龄点击该类目次数
    # t = train[['user_occupation_id','user_gender_id','user_star_level','user_age_level','cate2']]
    # t['gosa_cate_click'] = 1
    # t = t.groupby(['user_occupation_id','user_gender_id','user_star_level','user_age_level','cate2']).agg('sum').reset_index()
    # train = pd.merge(train,t,on=['user_occupation_id','user_gender_id','user_star_level','user_age_level','cate2'],how='left')
    # # 该年龄购买该类目次数
    # t = train[['user_occupation_id','user_gender_id','user_star_level','user_age_level','cate2','is_trade']]
    # t = t.groupby(['user_occupation_id','user_gender_id','user_star_level','user_age_level','cate2']).agg('sum').reset_index()
    # t = t.rename(columns={'is_trade':'gosa_cate_buy'})
    # train = pd.merge(train,t,on=['user_occupation_id','user_gender_id','user_star_level','user_age_level','cate2'],how='left')
    # # 该年龄购买该类别率
    # train['gosa_cate_rate'] = train['gosa_cate_buy']/train['gosa_cate_click']
    #
    # train[['gosa_cate_rate','user_occupation_id','user_gender_id','user_star_level','user_age_level','cate2']].to_csv('data/gosa_cate_feature1.csv',index=None)
    # # 该年龄星级
    # # 该年龄点击该类目次数
    # t = train[['user_star_level','user_age_level','cate2']]
    # t['star_age_cate_click'] = 1
    # t = t.groupby(['user_star_level','user_age_level','cate2']).agg('sum').reset_index()
    # train = pd.merge(train,t,on=['user_star_level','user_age_level','cate2'],how='left')
    # # 该年龄购买该类目次数
    # t = train[['user_star_level','user_age_level','cate2','is_trade']]
    # t = t.groupby(['user_star_level','user_age_level','cate2']).agg('sum').reset_index()
    # t = t.rename(columns={'is_trade':'star_age_cate_buy'})
    # train = pd.merge(train,t,on=['user_star_level','user_age_level','cate2'],how='left')
    # # 该年龄购买该类别率
    # train['star_age_cate_rate'] = train['star_age_cate_buy']/train['star_age_cate_click']
    #
    # train[['star_age_cate_rate','user_star_level','user_age_level','cate2']].to_csv('data/star_age_cate_feature1.csv',index=None)
    return train,test
#组合特征:用户-时间
def user_time_combien(train,test):
    # 该用户在当前时间点击次数
    t = train[['user_id','hour']]
    t['user_hour_click'] = 1
    t = t.groupby(['user_id','hour']).agg('sum').reset_index()
    train = pd.merge(train,t,on=['user_id','hour'],how='left')
    test = pd.merge(test,t,on=['user_id','hour'],how='left')
    # 该用户在当前时间购买次数
    t = train[['user_id','hour','is_trade']]
    t = t.groupby(['user_id','hour']).agg('sum').reset_index()
    t = t.rename(columns={'is_trade':'user_hour_buy'})
    train = pd.merge(train,t,on=['user_id','hour'],how='left')
    test = pd.merge(test,t,on=['user_id','hour'],how='left')
    # 该用户在当前时间的购买率
    train['user_hour_rate'] = train['user_hour_buy']/train['user_hour_click']
    test['user_hour_rate'] = test['user_hour_buy']/test['user_hour_click']
    return train,test

#组合特征:用户-品牌
def user_brand(train,test):
    #该性别购买该品牌次数
    t = train[['user_gender_id','item_brand_id','is_trade']]
    t = t.groupby(['user_gender_id','item_brand_id']).agg('sum').reset_index()
    t = t.rename(columns={'is_trade':'gender_brand_buy'})
    train = pd.merge(train,t,on=['user_gender_id','item_brand_id'],how='left')
    test = pd.merge(test,t,on=['user_gender_id','item_brand_id'],how='left')
    # 该性别点击该品牌次数
    t = train[['user_gender_id','item_brand_id']]
    t['gender_brand_click'] = 1
    t = t.groupby(['user_gender_id','item_brand_id']).agg('sum').reset_index()
    train = pd.merge(train,t,on=['user_gender_id','item_brand_id'],how='left')
    test = pd.merge(test,t,on=['user_gender_id','item_brand_id'],how='left')
    # 该性别购买率
    train['gender_brand_rate'] = train['gender_brand_buy']/train['gender_brand_click']
    test['gender_brand_rate'] = test['gender_brand_buy']/test['gender_brand_click']
    #
    # # 该星级
    # t = train[['user_star_level','item_brand_id','is_trade']]
    # t = t.groupby(['user_star_level','item_brand_id']).agg('sum').reset_index()
    # t = t.rename(columns={'is_trade':'star_brand_buy'})
    # train = pd.merge(train,t,on=['user_star_level','item_brand_id'],how='left')
    # test = pd.merge(test,t,on=['user_star_level','item_brand_id'],how='left')
    # # 该职业点击该品牌次数
    # t = train[['user_star_level','item_brand_id']]
    # t['star_brand_click'] = 1
    # t = t.groupby(['user_star_level','item_brand_id']).agg('sum').reset_index()
    # train = pd.merge(train,t,on=['user_star_level','item_brand_id'],how='left')
    # test = pd.merge(test,t,on=['user_star_level','item_brand_id'],how='left')
    # # 该职业购买率
    # train['star_brand_rate'] = train['star_brand_buy']/train['star_brand_click']
    # test['star_brand_rate'] = test['star_brand_buy']/test['star_brand_click']
    return train,test

#组合特征:用户-城市
def user_city(train,test):
    # 该用户点击该城市次数
    t = train[['user_id','item_city_id']]
    t['user_city_click'] = 1
    t = t.groupby(['user_id','item_city_id']).agg('sum').reset_index()
    train = pd.merge(train,t,on=['user_id','item_city_id'],how='left')
    test = pd.merge(test,t,on=['user_id','item_city_id'],how='left')
    # 该用户购买该城市次数
    t = train[['user_id','item_city_id','is_trade']]
    t = t.groupby(['user_id','item_city_id']).agg('sum').reset_index()
    t = t.rename(columns={'is_trade':'user_city_buy'})
    train = pd.merge(train,t,on=['user_id','item_city_id'],how='left')
    test = pd.merge(test,t,on=['user_id','item_city_id'],how='left')
    # 购买率
    train['user_city_rate'] = train['user_city_buy']/train['user_city_click']
    test['user_city_rate'] = test['user_city_buy']/test['user_city_click']
    return train,test

def user_prop_item(train,test):
    # 该职业点击该商品次数
    user_commodity = train
    t = user_commodity[['user_occupation_id','item_id']]
    t['occupation_item_click_total'] = 1
    t = t.groupby(['user_occupation_id','item_id']).agg(sum).reset_index()
    user_commodity = pd.merge(user_commodity,t,on=['user_occupation_id','item_id'],how='left')
    test = pd.merge(test,t,on=['user_occupation_id','item_id'],how='left')

    # 该职业购买该商品次数
    t = user_commodity[['user_occupation_id','item_id','is_trade']]
    t = t.groupby(['user_occupation_id','item_id']).agg(sum).reset_index()
    t.rename(columns={'is_trade':'occupation_item_click_buy_total'},inplace=True)
    user_commodity = pd.merge(user_commodity,t,on=['user_occupation_id','item_id'],how='left')
    test = pd.merge(test,t,on=['user_occupation_id','item_id'],how='left')
    # 该职业购买该商品率
    user_commodity['occupation_item_click_buy_rate'] = user_commodity['occupation_item_click_buy_total']/user_commodity['occupation_item_click_total']
    test['occupation_item_click_buy_rate'] = test['occupation_item_click_buy_total']/test['occupation_item_click_total']
    #
    # # 该年龄点击该商品次数
    # t = user_commodity[['user_age_level','item_id']]
    # t['age_item_click_total'] = 1
    # t = t.groupby(['user_age_level','item_id']).agg(sum).reset_index()
    # user_commodity = pd.merge(user_commodity,t,on=['user_age_level','item_id'],how='left')
    #
    # # 该年龄购买该商品次数
    # t = user_commodity[['user_age_level','item_id','is_trade']]
    # t = t.groupby(['user_age_level','item_id']).agg(sum).reset_index()
    # t.rename(columns={'is_trade':'age_item_click_buy_total'},inplace=True)
    # user_commodity = pd.merge(user_commodity,t,on=['user_age_level','item_id'],how='left')
    #
    # # 该年龄购买该商品率
    # user_commodity['age_item_click_buy_rate'] = user_commodity['age_item_click_buy_total']/user_commodity['age_item_click_total']
    # user_commodity[['age_item_click_buy_rate','age_item_click_buy_total','age_item_click_total','user_age_level','item_id']].to_csv('data/age_item_feature2.csv',index=None)
    #
    # # 该等级点击该商品次数
    # t = user_commodity[['user_star_level','item_id']]
    # t['star_item_click_total'] = 1
    # t = t.groupby(['user_star_level','item_id']).agg(sum).reset_index()
    # user_commodity = pd.merge(user_commodity,t,on=['user_star_level','item_id'],how='left')
    #
    # # 该等级购买该商品次数
    # t = user_commodity[['user_star_level','item_id','is_trade']]
    # t = t.groupby(['user_star_level','item_id']).agg(sum).reset_index()
    # t.rename(columns={'is_trade':'star_item_click_buy_total'},inplace=True)
    # user_commodity = pd.merge(user_commodity,t,on=['user_star_level','item_id'],how='left')
    #
    # # 该等级购买该商品率
    # user_commodity['star_item_click_buy_rate'] = user_commodity['star_item_click_buy_total']/user_commodity['star_item_click_total']
    # user_commodity[['star_item_click_buy_rate','star_item_click_buy_total','star_item_click_total','user_star_level','item_id']].to_csv('data/star_item_feature2.csv',index=None)
    #
    # # 该用户在多少家不同的店铺点击过该商品
    # t = user_commodity[['user_id','item_id','shop_id']]
    # t = t.drop_duplicates()
    # t['user_item_diff'] = 1
    # t = t.groupby(['user_id','item_id'])['user_item_diff'].agg('sum').reset_index()
    # user_commodity = pd.merge(user_commodity,t,on=['item_id','user_id'],how='left')
    #
    # # 该年龄职业点击该商品次数
    # t = user_commodity[['item_id','user_age_level','user_occupation_id']]
    # t['age_occupation_click_total'] = 1
    # t = t.groupby(['user_age_level','user_occupation_id','item_id']).agg('sum').reset_index()
    # user_commodity = pd.merge(user_commodity,t,on=['user_age_level','user_occupation_id','item_id'],how='left')
    # # 该年龄职业购买该商品数
    # t = user_commodity[['item_id','user_age_level','user_occupation_id','is_trade']]
    # t = t.groupby(['user_age_level','user_occupation_id','item_id']).agg('sum').reset_index()
    # t = t.rename(columns={'is_trade':'age_occupation_click_buy_total'})
    # user_commodity = pd.merge(user_commodity,t,on=['user_age_level','user_occupation_id','item_id'],how='left')
    # # 该年龄职业购买该商品率
    # user_commodity['age_occupation_click_buy_rate'] = user_commodity['age_occupation_click_buy_total']/user_commodity['age_occupation_click_total']
    # user_commodity[['age_occupation_click_total','age_occupation_click_buy_total','age_occupation_click_buy_rate','item_id','user_age_level','user_occupation_id']].to_csv('data/age_occupation_item_feature2.csv',index=None)
    #
    # # 该年龄星级点击该商品次数
    # t = user_commodity[['item_id','user_age_level','user_star_level']]
    # t['age_star_click_total'] = 1
    # t = t.groupby(['user_age_level','user_star_level','item_id']).agg('sum').reset_index()
    # user_commodity = pd.merge(user_commodity,t,on=['user_age_level','user_star_level','item_id'],how='left')
    return user_commodity,test

#商品额外特征
def comm_extra(data):
    # 商品额外特征
    # 该商品当天被点击次数
    t = data[['item_id']]
    t['item_today_click'] = 1
    t = t.groupby('item_id').agg('sum').reset_index()
    data = pd.merge(data,t,on='item_id',how='left')
    # 该商品当天被多少不同的用户点击
    t = data[['item_id','user_id']]
    t = t.drop_duplicates()
    t['item_diffuser_click'] = 1
    t = t.groupby(['item_id'])['item_diffuser_click'].agg('sum').reset_index()
    # # 该商品的点击率
    # train['item_today_rate'] = train['item_today_click']/train.shape[0]
    data = pd.merge(data,t,on='item_id',how='left')
    return data

#商品性别当天特征
def comm_gender(data):
    # 该商品当天被某一个性别点击次数
    t = data[['item_id','user_gender_id']]
    t['item_gender_today_click'] = 1
    t = t.groupby(['item_id','user_gender_id'])['item_gender_today_click'].agg('sum').reset_index()
    data = pd.merge(data,t,on=['item_id','user_gender_id'],how='left')
    return data

#用户当天额外特征
def user_extra(data):
    train = data
    #用户当天点击次数
    t = train[['user_id']]
    t['user_today_click'] = 1
    t = t.groupby('user_id').agg('sum').reset_index()
    train = pd.merge(train,t,on='user_id',how='left')
    # 用户当天点击商家数目
    t = train[['user_id','shop_id']]
    t.drop_duplicates(inplace=True)
    t['user_today_num']=1
    t = t.groupby('user_id')['user_today_num'].agg('sum').reset_index()
    train = pd.merge(train,t,on='user_id',how='left')
    # 用户当天点击不同商品数目
    t = train[['user_id','item_id']]
    t.drop_duplicates(inplace=True)
    t['user_today_item_num']=1
    t = t.groupby('user_id')['user_today_item_num'].agg('sum').reset_index()
    train = pd.merge(train,t,on='user_id',how='left')
    # 用户当天点击种类数目
    t = train[['user_id','second_cate']]
    t = t.drop_duplicates()
    t['user_cate_today_sum'] = 1
    t = t.groupby('user_id')['user_cate_today_sum'].agg('sum').reset_index()
    train = pd.merge(train,t,on='user_id',how='left')
    # 用户当天点击品牌数目
    t = train[['user_id','item_brand_id']]
    t = t.drop_duplicates()
    t['user_brand_today_sum'] = 1
    t = t.groupby('user_id')['user_brand_today_sum'].agg('sum').reset_index()
    train = pd.merge(train,t,on='user_id',how='left')
    return train

#商家当天额外特征
def shop_extra(data):
    # 商家当天被点击次数
    train = data
    t  = train[['shop_id']]
    t['shop_today_beclick'] = 1
    t = t.groupby('shop_id').agg('sum').reset_index()
    train = pd.merge(train,t,on='shop_id',how='left')

    # 用户点击该商家次数占用户当天点击比值
    t = train[['user_id','shop_id']]
    t['user_shop_today'] = 1
    t = t.groupby(['user_id','shop_id']).agg('sum').reset_index()
    train = pd.merge(train,t,on=['user_id','shop_id'],how='left')
    train['user_shop_rate'] = train['user_shop_today']/train['user_today_click']
    #用户点击该商家次数占商家当天点击比值
    train['shop_user_rate'] = train['user_shop_today']/train['shop_today_beclick']
    return train

#品牌额外特征
def brand_extra(data):
    train =data
    t = train[['item_brand_id']]
    t['brand_today_click'] =1
    t = t.groupby('item_brand_id').agg('sum').reset_index()
    train = pd.merge(train,t,on='item_brand_id',how='left')
    return train

#用户当天点击该等级的商品
def user_click_comm_level(data):
    train = data

    # 用户当天点击该价格等级的数目
    t = train[['user_id','item_price_level']]
    t['user_price_ctoday'] = 1
    t = t.groupby(['user_id','item_price_level']).agg('sum').reset_index()
    train = pd.merge(train,t,on=['user_id','item_price_level'],how='left')
    # 用户点击该价格等级商品占用户点击比
    train['user_price_tcrate'] = train['user_price_ctoday']/train['user_today_click']
    # # 用户点击改价格等级商品占该价格等级点击比

    # ,'price_user_tcrate'
    # 用户当天点击该收藏等级的数目
    t = train[['user_id','item_collected_level']]
    t['user_collected_ctoday'] = 1
    t = t.groupby(['user_id','item_collected_level']).agg('sum').reset_index()
    train = pd.merge(train,t,on=['user_id','item_collected_level'],how='left')
    # 用户点击该价格等级商品占用户点击比
    train['user_collected_tcrate'] = train['user_collected_ctoday']/train['user_today_click']
    # 用户点击该价格等级商品占价格等级点击比
    t = train[['item_collected_level']]
    t['collected_num'] = 1
    t = t.groupby('item_collected_level').agg('sum').reset_index()
    train = pd.merge(train,t,on='item_collected_level',how='left')
    train['collected_user_tcrate'] = train['user_collected_ctoday']/train['collected_num']

    # 用户当天点击该sale等级的数目
    t = train[['user_id','item_sales_level']]
    t['user_sales_ctoday'] = 1
    t = t.groupby(['user_id','item_sales_level']).agg('sum').reset_index()
    train = pd.merge(train,t,on=['user_id','item_sales_level'],how='left')
    # 用户点击该价格等级商品占用户点击比
    train['user_sales_tcrate'] = train['user_sales_ctoday']/train['user_today_click']
    # 用户点击该价格等级商品占价格等级点击比
    t = train[['item_sales_level']]
    t['sales_num'] = 1
    t = t.groupby('item_sales_level').agg('sum').reset_index()
    train = pd.merge(train,t,on='item_sales_level',how='left')
    train['sales_user_tcrate'] = train['user_sales_ctoday']/train['sales_num']

    return train
