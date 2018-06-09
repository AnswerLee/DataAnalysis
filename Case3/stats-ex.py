# -*- coding: utf-8 -*-
"""
Created on Sat Jun  9 09:45:45 2018

@author: AnswerLee
"""

#%%
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy import stats
from scipy.stats import norm, skew 
import os
os.chdir(r'D:\give_me_five\githome\DataAnalysis\Case3')
#%%

hp_predict = pd.read_csv('./dataset/sndHsPr.csv',encoding='gbk')
#%%
# 可以看出房价是的分布是右偏的
sns.distplot(hp_predict.price,bins=20,fit=norm)
# 使用probplot函数
fig = plt.figure()
stats.probplot(hp_predict.price, plot=plt)
plt.show()
#%%
# 描述性统计分析
# 单变量看均值、中位数、上四分位和下四分位数
# 可以看出中位数是57473，均值是61151，分布右偏
hp_predict.price.describe()
#%%
# 偏度值大于0 ，右偏
hp_predict.price.skew()
#%%
# 线性回归模型的使用假设因变量服从正态分布
# 对price进行log变换,可以看出来是基本符合正态分布的
hp_predict['price_ln'] = np.log1p(hp_predict.price)
sns.distplot(hp_predict.price_ln,bins=10,fit=norm)
# 使用probplot函数  查看
fig = plt.figure()
stats.probplot(hp_predict.price_ln,plot=plt)
plt.show()
#%%
# 自变量自身分布分析
# 看看样本每个区域的分布情况
# 各个区样本数量
sns.countplot('dist',data=hp_predict)
#%%
'''
dist:分类变量   方差分析 ANOVA
roomnum:分类变量   方差分析 ANOVA
halls:分类变量   方差分析 ANOVA
AREA:连续变量   相关分析
floor:分类变量   方差分析 ANOVA
subway:分类变量   方差分析 ANOVA
school:分类变量   方差分析  ANOVA
price:连续变量   
'''
#%%
# 查看缺失值, 每一列都没有缺失值，不用填充
hp_predict.info()
#%%
'''
查看有几个区
共有{'chaoyang', 'haidian', 'fengtai', 'shijingshan', 'dongcheng', 'xicheng'}
6个区
'''
print(set(hp_predict.dist))
'''
查看每个区房屋的均价，差别较大
'''
hp_predict.groupby(by=['dist'])['price'].mean().plot(kind='bar')
#%%
'''
查看每个区房屋的盒须图，中位数、上下四分位数都差别较大
'''
sns.boxplot('dist','price',data=hp_predict)
#%%
# 方差分析
sm.stats.anova_lm(ols('price_ln ~ C(dist)',data=hp_predict).fit())
#%%
'''
查看roomnum共有几种取值,
共有{1, 2, 3, 4, 5}5种取值
'''
print(set(hp_predict.roomnum))
'''
查看不同房间数的房屋均价,差异较小
'''
hp_predict.groupby(['roomnum'])['price'].mean().plot(kind='bar')
#%%
'''
查看房间数、房屋价格的盒须图，中位数差异较小
下四分位数几乎一致，上四分位数差异较小
'''
sns.boxplot('roomnum','price',data=hp_predict)
#%%
# 方差分析
sm.stats.anova_lm(ols('price ~ C(roomnum)',data=hp_predict).fit())
#%%
'''
查看halls共有几种取值,
共有{0, 1, 2, 3}4种取值
'''
print(set(hp_predict.halls))
'''
查看不同厅数的房屋均价，存在差异，较小，
厅数越多，房屋单价均值越低
'''
hp_predict.groupby(['halls'])['price'].mean().plot(kind='bar')
#%%
'''
查看厅的个数、房屋单价的盒须图，
中位数、下四分位差异不明显，上四分位差异明显
厅数越多，中位数越低
'''
sns.boxplot('halls','price',data=hp_predict)
#%%
# 方差分析
sm.stats.anova_lm(ols('price_ln ~ C(halls)',data=hp_predict).fit())
#%%
'''
查看房屋面积和房屋单价的散点图, 
随着面积增加单价有下降的趋势，但是不是很明显
'''
hp_predict.plot.scatter('AREA','price')
#%%
'''
查看房屋面积和房屋单价的log变换的散点图，
随着面积增加，price_log的取值趋于集中
'''
hp_predict.plot.scatter('AREA','price_ln')
#%%
'''
房屋面积和房屋单价的相关性分析,默认pearson,数值为-0.07395
'''
hp_predict[['AREA','price']].corr()
#%%
'''
查看floor的取值个数，
共{'low','middle','high'}三个取值
'''
print(set(hp_predict.floor))
'''
查看不同楼层的房屋单价均值,
中楼层偏高、低楼层次之、高楼层偏低
'''
hp_predict.groupby(['floor'])['price'].mean().plot(kind='bar')
#%%
'''
查看不同楼层、房屋单价的盒须图，
中位数的趋势一致  中、低、高
'''
sns.boxplot('floor','price',data=hp_predict)
#%%
# 方差分析
sm.stats.anova_lm(ols('price_ln ~ C(floor)',data=hp_predict).fit())
#%%
'''
查看subway有几种取值
共{0,1}两种取值，代表有、无地铁
'''
print(set(hp_predict.subway))
'''
查看地铁和房屋单价均值的关系,
有较大差异，有地铁明显比无地铁高
'''
hp_predict.groupby(['subway'])['price'].mean().plot(kind='bar')
#%%
'''
查看地铁、房屋单价的盒须图，
有地铁的房屋 中位数、上四分位数 明显高于无地铁，
'''
sns.boxplot('subway','price',data=hp_predict)
#%%
# 方差分析  
sm.stats.anova_lm(ols('price_ln ~ C(subway)',data=hp_predict).fit())
#%%
'''
查看school的取值
共{0,1}两种取值，即是或者不是学区房
'''
print(set(hp_predict.school))
'''
查看school 和房屋单价均值的关系,
学区房的单价均值明显高于非学区房
'''
hp_predict.groupby(['school'])['price'].mean().plot(kind='bar')
#%%
'''
查看school 、房屋单价的盒须图，
学区房的中位数、上四分位数明显高于非学区房
'''
sns.boxplot('school','price',data=hp_predict)
#%%
# 方差分析  pr值为0.0
sm.stats.anova_lm(ols('price_ln ~ C(school)',data=hp_predict).fit())
#%%
# 多变量方差分析
# roomnum的PR值是0.95  这么大，可以认为不显著？
sm.stats.anova_lm(ols('price ~ C(dist) + C(school) + C(subway) + C(roomnum) + C(floor) + C(halls)',data=hp_predict).fit())
#%%
'''
经过以上变量的分析
明显相关的自变量有dist、floor、subway、school
相关，但是差异不明显的有roomnum、halls
唯一一个连续性变量AREA与房屋单价负相关


假设有一家三口，父母为了能让孩子在东城区上学，想买一套邻近地铁的两居室，面积是70平方米，中层楼层，
dist='dongcheng'
subway=1
roomnum=2
school=1
floor='middle'
AREA=70.0

预测价格为 79342元人民币
'''


test_0 = pd.DataFrame({'dist':'dongcheng','subway':1,'roomnum':2,'halls':0,'floor':'middle','AREA':70.0,'school':1},
                       index=[0],columns=['dist','roomnum','floor','subway','school','halls','AREA'])
test_1 = pd.DataFrame({'dist':'dongcheng','subway':1,'roomnum':2,'halls':1,'floor':'middle','AREA':70.0,'school':1},
                       index=[0],columns=['dist','roomnum','floor','subway','school','halls','AREA'])
test_2 = pd.DataFrame({'dist':'dongcheng','subway':1,'roomnum':2,'halls':2,'floor':'middle','AREA':70.0,'school':1},
                       index=[0],columns=['dist','roomnum','floor','subway','school','halls','AREA'])
#%%
# 多元线性回归变量筛选
'''forward select'''
def forward_select(data, response):
    remaining = set(data.columns)
    remaining.remove(response)
    selected = []
    current_score, best_new_score = float('inf'), float('inf')
    while remaining:
        aic_with_candidates=[]
        for candidate in remaining:
            formula = "{} ~ {}".format(
                response,' + '.join(selected + [candidate]))
            aic = ols(formula=formula, data=data).fit().aic
            aic_with_candidates.append((aic, candidate))
        aic_with_candidates.sort(reverse=True)
        best_new_score, best_candidate=aic_with_candidates.pop()
        if current_score > best_new_score: 
            remaining.remove(best_candidate)
            selected.append(best_candidate)
            current_score = best_new_score
            print ('aic is {},continuing!'.format(current_score))
        else:        
            print ('forward selection over!')
            break
            
    formula = "{} ~ {} ".format(response,' + '.join(selected))
    print('final formula is {}'.format(formula))
    model = ols(formula=formula, data=data).fit()
    return(model)
# 建立回归模型， halls=0
data_for_select = hp_predict[['dist','subway','school','roomnum','halls','floor','AREA','price']]
fs_model = forward_select(data_for_select,response='price')
print(fs_model.rsquared)
print(fs_model.predict(test_0))
#%%
# 建立回归模型，halls = 1
print(fs_model.predict(test_1))
#%% 
# 建立回归模型，halls = 2
print(fs_model.predict(test_2))
#%%
# 建立回归模型，对因变量取对数  halls=0
data_for_select_1 = hp_predict[['dist','subway','school','roomnum','halls','floor','AREA','price_ln']]
fs_ln_model = forward_select(data_for_select_1,response='price_ln')
print(fs_ln_model.rsquared)
print(np.expm1(fs_ln_model.predict(test_0)))
#%%
# 建立回归模型，对因变量取对数。  halls=1
print(np.expm1(fs_ln_model.predict(test_1)))
#%%
# 建立回归模型，对因变量取对数 halls=2
print(np.expm1(fs_ln_model.predict(test_2)))
#%%







