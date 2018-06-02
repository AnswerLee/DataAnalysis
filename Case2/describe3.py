# -*- coding: utf-8 -*-
"""
Created on Thu May 31 22:17:15 2018

@author: AnswerLee
"""

#%%
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

os.chdir(r'D:\give_me_five\githome\DataAnalysis\Case2')

from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei'] # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题
#%%
accounts = pd.read_csv('./dataset/loan/accounts.csv',encoding='gbk')
card = pd.read_csv('./dataset/loan/card.csv',encoding='gbk')
clients = pd.read_csv('./dataset/loan/clients.csv',encoding='gbk')
disp = pd.read_csv('./dataset/loan/disp.csv',encoding='gbk')
district = pd.read_csv('./dataset/loan/district.csv',encoding='gbk')
loans = pd.read_csv('./dataset/loan/loans.csv',encoding='gbk')
order = pd.read_csv('./dataset/loan/order.csv',encoding='gbk')
#%%
trans = pd.read_csv('./dataset/loan/trans.csv',encoding='gbk')
#%%
# 连接clients和disp两张表
clients_cards = pd.merge(disp,clients,on='client_id',how='left')
clients_cards = pd.merge(card,clients_cards,on='disp_id',suffixes=('_card','_disp'))
#%%
# 不同类型卡的持卡人在办卡时的平均年龄对比
# 两个分类变量之间的关系
# 青年卡平均年龄在18岁左右，这么年轻？我一直认为青年是18-30之间
# 普通卡和金卡都是40多岁，金卡比普通卡稍微高一点
client_age = clients_cards[['birth_date','issued','type_card']]
client_age['b_date'] = pd.to_datetime(client_age.birth_date,format='%Y-%m-%d').map(lambda x:x.year)
client_age['i_date'] = pd.to_datetime(client_age.issued,format='%Y-%m-%d').map(lambda x:x.year)
client_age['issued_age'] = client_age.i_date - client_age.b_date
client_age.groupby(['type_card'])['issued_age'].mean().sort_values().plot(kind='bar')
#%%
# 查看普通卡和金卡中年龄分布情况
# 普通卡持有人数比金卡持有人数多很多
# 年轻人中真有土豪金呀。。。而且25-35岁之间金卡数量稍微多一些
client_age[client_age.type_card=='普通卡']['issued_age'].hist(bins=10,color='b')
client_age[client_age.type_card=='金卡']['issued_age'].hist(bins=10,color='g')
#%%
# 查看青年卡中年龄分布，12岁就能办卡了？我一直以为18周岁。。。
client_age[client_age.type_card=='青年卡']['issued_age'].hist(bins=10,color='g')
#%%
# 查看金卡、普通卡、青年卡的数量
# 普通卡肯定最多
client_age.groupby(['type_card'])['type_card'].count().sort_values().plot(kind='barh')
#%%
# 查看不同性别的不同类型卡持有人数对比
pd.crosstab(clients_cards.sex,clients_cards.type_card).plot(kind='bar')
#%%
def toInt(x):
    tmp = x.replace('$','')
    tmp = tmp.replace(',','')
    return int(tmp)
#%%
# 连接trans、accounts、disp、card四张表
trans_cards = pd.merge(trans,accounts,on='account_id',how='left',suffixes=('_trans','_acnt'))
trans_cards = pd.merge(trans_cards,disp,on='account_id',how='left',suffixes=('_trans','_disp'))
trans_cards = pd.merge(card,trans_cards,on='disp_id',how='left')
trans_cards.head()
#%%
# 数据清洗和处理
trans_cards.issued = pd.to_datetime(trans_cards.issued,format='%Y-%m-%d')
trans_cards.date_trans = pd.to_datetime(trans_cards.date_trans,format='%Y-%m-%d')
trans_cards.date_acnt = pd.to_datetime(trans_cards.date_acnt,format='%Y-%m-%d')
trans_cards.balance = trans_cards.balance.map(toInt)
trans_cards.amount = trans_cards.amount.map(toInt)
type_trans_map = {'借':'out','贷':'income'}
trans_cards.type_trans = trans_cards.type_trans.map(type_trans_map)
trans_cards.head()
#%%
# 过滤出办卡前一年的交易信息
import datetime
trans_info = trans_cards[['type','issued','type_trans','date_trans','amount','balance']]
trans_info = trans_info[(trans_info.date_trans < trans_info.issued) & (trans_info.date_trans > trans_info.issued - datetime.timedelta(days=365))]
trans_info.head()
#%%
# 不同类型卡的持卡人办卡前一年的平均账户余额对比
trans_info.groupby(['type'])['balance'].mean().sort_values().plot(kind='barh')
#%%
# 不同类型卡的持卡人办卡前一年的平均收入和平均支出对比
pd.pivot_table(trans_info,values='amount',index='type',columns='type_trans',aggfunc='mean').plot(kind='bar')

#%%