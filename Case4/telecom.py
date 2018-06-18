# -*- coding: utf-8 -*-
"""
Created on Sat Jun 16 16:35:21 2018

@author: AnswerLee
"""

#%%
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
import os
os.chdir(r'D:\give_me_five\githome\DataAnalysis\Case4')
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决画图时负号显示为方框问题
plt.rcParams['font.sans-serif'] = ['SimHei']   #指定默认字符集
#%%
'''
主要变量预测如下：
subscriberID="个人客户的ID"

churn="是否流失：1=流失";

Age="年龄"

incomeCode="用户居住区域平均收入的代码"

duration="在网时长"

peakMinAv="统计期间内最高单月通话时长"

peakMinDiff="统计期间结束月份与开始月份相比通话时长增加数量"

posTrend="该用户通话时长是否呈现出上升态势：是=1"

negTrend="该用户通话时长是否呈现出下降态势：是=1"

nrProm="电话公司营销的数量"

prom="最近一个月是否被营销过：是=1"

curPlan="统计时间开始时套餐类型：1=最高通过200分钟；2=300分钟；3=350分钟；4=500分钟"

avPlan="统计期间内平均套餐类型"

planChange="统计期间是否更换过套餐：1=是"

posPlanChange="统计期间是否提高套餐：1=是"

negPlanChange="统计期间是否降低套餐：1=是"

call_10086="拨打10086的次数"
'''
dat0 = pd.read_csv('./dataset/telecom_churn.csv')
dat0.head()
#%%
# 2.2.1 两变量分析
# 检测该用户通话时长是否呈现上升态势(posTrend)对流失(churn)是否有侧价值
# posTrend 变量取值
print(dat0.posTrend.unique())
# churn 变量取值
print(dat0.churn.unique())
# 两个类别型变量  使用交叉表分析
dat1 = pd.crosstab(dat0.churn,dat0.posTrend)
# 在流失客户中，通话时长呈上升趋势和非上升趋势的比例  35.5% : 64.5%
# 在非流失客户中，通话时长呈上升趋势和非上升趋势的比例 43% ：57% 
print(dat1.div(dat1.sum(1),axis=0))
# 在呈上升趋势客户中，流失和非流失比例是 33%:67%
# 在呈非上升趋势客户中，流失和费流失比例  是 54.4% : 45.6%
print(dat1.div(dat1.sum(0),axis=1))
#  最终得出结论，通话时长是否呈上升态势对流失有影响
dat1.plot(kind='bar')
#%%
# 2.2.2  逻辑回归模型
# 在网时长  duration  数值型变量
# 单变量分析，描述性统计分析
# 最大值73 最小值2  中位数小于平均值  右偏
print(dat0.duration.agg(['mean','median','std','min','max']))
print(dat0.duration.describe().T)
fig = plt.figure(figsize=(8,6))
dat0.duration.hist(bins=20)
# 流失客户平均在网时长，明显小于非流失客户， 7:20
dat2 = dat0.groupby(['churn'])['duration'].mean()
fig = plt.figure(figsize=(8,6))
dat2.plot(kind='bar')
plt.xlabel('是否流失')
plt.ylabel('平均入网时长')
plt.title('流失客户和非流失客户的平均在网时长')
#%%
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc,roc_auc_score,roc_curve
from sklearn.metrics import accuracy_score,recall_score,confusion_matrix

dat2x = dat0.duration
dat2y = dat0.churn

# 拆分数据集，分为训练集和测试集
X_train,X_test,y_train,y_test = train_test_split(dat2x,dat2y,test_size=0.3)
#%%
#  训练模型
lr = LogisticRegression()
lr.fit(X_train.values.reshape(-1,1),y_train)
# 预测概率
y_test_proba = lr.predict_proba(X_test.values.reshape(-1,1))
# 预测流失  阈值为0.5
y_test_s = pd.Series(y_test_proba[:,1])
y_test_s = y_test_s.map(lambda x: 1 if x >= 0.5 else 0)
# 准确率
print('accuracy score: ' , round(accuracy_score(y_test,y_test_s.values),3))
# 召回率
print('recall score: %.3f' % recall_score(y_test,y_test_s.values))
# 混淆矩阵
confusion_matrix(y_test,y_test_s)
# 衡量指标
fpr,tpr,thresholds = roc_curve(y_test,y_test_proba[:,-1])
# auc  0.864  是不是有点高了？
print('auc : %.3f' % auc(fpr,tpr))
# ROC 曲线
fig = plt.figure(figsize=(8,6))
plt.plot(fpr,tpr,'k--',label='ROC (area = %.3f)' % auc(fpr,tpr))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(' Roc Curve')
plt.legend()
plt.show()
#%%
# 2.2.3其他备选变量如下
'''
gender : 二值分类变量，使用独热编码
AGE ： 连续型变量， 考虑使用最小最大值缩放
edu_class :  当做连续变量处理，使用最大最小值缩放
incomeCode : 连续型变量，考虑使用最大最小值缩放
duration ： 连续变量，考虑使用最大最小值缩放
feton  : 二值分类变量，使用独热编码
peakMinAv : 连续型变量，考虑使用最大最小值缩放
peakMinDiff : 连续型变量，考虑使用最大最小值缩放
posTrend : 二值分类变量，使用独热编码
negTrend : 二值分类变量，使用独热编码
nrProm : 连续变量，考虑使用最大最小值缩放
prom ： 二值分类变量 使用独热编码
curPlan :  当做连续变量来处理，使用最大最小值缩放
avgplan : 当做连续变量来处理，使用最大最小值缩放
planChange : 连续变量，使用最大最小值缩放
posPlanChange : 二值分类变量，使用独热编码
negPlanChange ： 二值分类变量，使用独热编码
call_10086 : 二值分类变量，使用独热编码
'''
# 处理分类变量
dat3 = dat0.drop(['subscriberID'],axis=1).copy()
#%%
categorical_cols = ['gender','feton','posTrend','negTrend',
        'prom','posPlanChange','negPlanChange','call_10086']
for col in categorical_cols:
    dat3[col] = dat3[col].astype('category',categories=list(dat3[col].unique()))
    gen_dum = pd.get_dummies(dat3[col],prefix=col)
    dat3 = dat3.join(gen_dum).drop([col],axis=1)
#%%
# 处理连续值变量, 保持数值范围在同一个数量级
from sklearn.preprocessing import MinMaxScaler
mmScaler = MinMaxScaler()
numerical_cols = ['AGE','edu_class','incomeCode','duration',
                  'peakMinAv','peakMinDiff','nrProm','curPlan',
                  'avgplan','planChange']

for col in numerical_cols:
    dat3[col] = mmScaler.fit_transform(dat3[col].values.reshape(-1,1))
#%%

# 向前逐步回归选择变量
'''forward select'''
def forward_select(data, response):
    remaining = set(data.columns)
    remaining.remove(response)
    selected = []
    current_score, best_new_score = float('inf'), float('inf')
    while remaining:
        aic_with_candidates=[]
        for candidate in remaining:
            aic = smf.Logit(data[response],data[selected+[candidate]]).fit().aic
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
    return selected
selected_cols = forward_select(dat3,'churn')
dat4 = dat3[selected_cols + ['churn']]
#%%
# 检查模型的膨胀系数
from statsmodels.stats.outliers_influence import variance_inflation_factor
def FuncVif(df):
    vif = pd.DataFrame({'features':df.columns})
    vif['vifValue'] = [variance_inflation_factor(df.values,i) for i in range(df.shape[1])]
    vif = vif.sort_values(by='vifValue',ascending=False)
    return vif
FuncVif(dat4)
#%%
# 删除vif值最高的planChange
dat4 = dat4.drop(['planChange'],axis=1)
FuncVif(dat4)
#%%
# 继续删除 peakMinDiff
dat4 = dat4.drop(['peakMinDiff'],axis=1)
FuncVif(dat4)
#%%
# 继续删除 nrProm
dat4 = dat4.drop(['nrProm'],axis=1)
FuncVif(dat4)
# 此时vif都小于10
#%%
# 训练模型
def FuncScore(y_true,y_score):
    
    y_pred = y_score.map(lambda x: 1 if x >= 0.5 else 0)
    # 准确率
    print('accuracy score: ' , round(accuracy_score(y_true,y_pred.values),3))
    # 召回率
    print('recall score: %.3f' % recall_score(y_true,y_pred.values))
    # 混淆矩阵
    confusion_matrix(y_true,y_pred)
    # 衡量指标
    fpr,tpr,thresholds = roc_curve(y_true,y_score)
    # auc  
    print('auc : %.3f' % auc(fpr,tpr))
    # ROC 曲线
    plt.plot(fpr,tpr,'k--',label='ROC (area = %.3f)' % auc(fpr,tpr))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(' Roc Curve')
    plt.legend()
    plt.show()
    
dat4x = dat4.drop(['churn'],axis=1)
dat4y = dat4.churn

X_train,X_test,y_train,y_test = train_test_split(dat4x,dat4y,test_size=0.3)

model4 = smf.Logit(y_train,X_train).fit()
y_test_proba_4 = model4.predict(X_test)

# auc 面积高达0.913  是不是有问题？
FuncScore(y_test,y_test_proba_4)
#%%
# 2.2.4  使用LASSO 和Ridge
# 查看下样本比例，4:3 ，接近1：1
print(dat3.churn.value_counts())
# 使用网格搜索交叉验证
# 训练集 、 测试集拆分
from sklearn.model_selection import GridSearchCV

dat5x = dat3.drop(['churn'],axis=1)
dat5y = dat3.churn

X_train,X_test,y_train,y_test = train_test_split(dat5x,dat5y,test_size=0.3,random_state=1024)

param_grid = {'C':[0.10,0.15,0.5,0.7,1.0,1.2,1.5,1.8,2.0]}
#%%
# LASSO
clf = LogisticRegression(penalty='l1')
# 三折交叉验证
grid_search = GridSearchCV(clf,param_grid,cv=5,scoring='roc_auc')
grid_search.fit(X_train,y_train)
print(grid_search.best_estimator_)
# 使用最优惩罚系数建立模型
lasso = LogisticRegression(penalty='l1',solver='liblinear',C=2.0)
lasso.fit(X_train,y_train)
y_test_proba_l = lasso.predict_proba(X_test)
y_test_proba_l = pd.Series(y_test_proba_l[:,1])
# 查看准确率，召回率  准确率0.828，召回率0.8，roc0.92  有点高，不太靠谱吧？
FuncScore(y_test,y_test_proba_l)
# 查看每一个特征的系数
coef = pd.DataFrame()
coef['feature'] = X_test.columns
coef['value'] = lasso.coef_.ravel()
# 查看膨胀系数  
cols = coef[coef.value != 0 ]['feature']
FuncVif(dat3[cols])
#%%
# Ridge  逻辑回归默认使用L2正则化
clf = LogisticRegression()
grid_search = GridSearchCV(clf,param_grid,cv=5,scoring='roc_auc')
grid_search.fit(X_train,y_train)
print(grid_search.best_estimator_)

# 使用最优惩罚系数建立模型
ridge = LogisticRegression(penalty='l2',C=2.0)
ridge.fit(X_train,y_train)
y_test_proba_r = ridge.predict_proba(X_test)
y_test_proba_r = pd.Series(y_test_proba_r[:,1])
FuncScore(y_test,y_test_proba_r)
# 查看每一个特征的系数
coef = pd.DataFrame()
coef['feature'] = X_test.columns
coef['value'] = ridge.coef_.ravel()
# 查看膨胀系数
cols = coef[coef.value != 0 ]['feature']
FuncVif(dat3[cols])
#%%
