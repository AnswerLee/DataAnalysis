# -*- coding: utf-8 -*-
"""
Created on Wed May 30 21:54:40 2018

@author: AnswerLee
"""
#%%
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
#%%
from pylab import mpl
mpl.rcParams["font.sans-serif"] = ["SimHei"]#指定默认字体
mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题
#%%
import os
os.chdir(r'D:\give_me_five\githome\DataAnalysis\Case2')
#%%
auto_ins = pd.read_csv("./dataset/auto_ins.csv",encoding='gbk')
auto_ins.head()
#%%
auto_ins["loss_flag"] = auto_ins.Loss.map(lambda x:x if x == 0 else 1).astype('Int64')
auto_ins.head()
#%%
#对loss_flag单分类变量进行分析  频数
auto_ins.loss_flag.value_counts().plot(kind="bar")
#%%
#饼状图   出现次数占总次数的28.5%
auto_ins.loss_flag.value_counts().plot(kind="pie",autopct='%.1f',explode=(0,0.1))
#%%
vc1 = auto_ins.loss_flag.value_counts()
type(vc1)
#%%
# 两个分类变量之间的关系  性别和loss_flag
# 查看男性和女性loss为0,1的数量
# 男性出险次数较多， 毕竟男性开车多，基数大
loss_gd = pd.crosstab(auto_ins.Gender,auto_ins.loss_flag)
loss_gd.plot(kind="bar")
#%%
# 查看男性和女性loss为0,1的比例
# 女性出险比例高于男性，  女性开车还是不太稳呀
loss_gd.div(loss_gd.sum(1),axis=0).plot(kind="bar",stacked=True)
#%%
from stack2dim import *
stack2dim(auto_ins,"Gender","loss_flag")
#%%
# 查看婚姻状态和loss_flag之间的关系
# 已婚出险次数远高于未婚，有可能是单身买车相对少
loss_me = pd.crosstab(auto_ins.Marital,auto_ins.loss_flag)
loss_me.plot(kind='bar')
#%%
# 查看婚姻状态和loss_flag 比例关系
# 未婚出险比例较高，单身 年轻开车猛
loss_me.div(loss_me.sum(1),axis=0).plot(kind="bar",stacked=True)
#%%
stack2dim(auto_ins,"Marital","loss_flag")
#%%
# 查看汽车所有权人和出险的关系
# 从出险次数、出险比例上，私人 > 公司 > 政府
loss_ow = pd.crosstab(auto_ins.Owner,auto_ins.loss_flag)
loss_ow.plot(kind="bar")
#%%
loss_ow.div(loss_ow.sum(1),axis=0).plot(kind="bar",stacked=True)
#%%
stack2dim(auto_ins,"Owner","loss_flag")
#%%
# 查看年龄和出险次数的关系
auto_ins_loss = auto_ins[auto_ins.loss_flag==1]
auto_ins_loss.head()
#%%
auto_ins_loss.Age.mean()
#%%
auto_ins_loss.Age.min()
#%%
auto_ins_loss.Age.max()
#%%
auto_ins_loss.Age.median()
#%%
auto_ins_loss.Age.std()
#%%
auto_ins_loss.Age.describe()
#%%
auto_ins_loss.Age.quantile([0.01,0.5,0.99])
#%%
auto_ins_loss.Age.skew()
#%%
# 可以看出30-45之间年龄段的出现次数较高
# 这个阶段正处于中年，车是必备
auto_ins_loss.Age.hist(bins=10)
#%%
sns.boxplot(x=auto_ins.loss_flag,y=auto_ins.Age)
#%%
#将年龄分为几段25岁及以下，26-35,36-45,46-55,56岁及以上
loss_age = auto_ins[['Age','loss_flag']]
bins = pd.IntervalIndex.from_tuples([(0,25),(25,35),(35,45),(45,55),(55,100)],closed='left')
age_label = pd.cut(auto_ins.Age.values,bins=bins,right=False)
loss_age['age_label'] = age_label
loss_age.head()
#%%
# 查看每个年龄段的出险次数和比例
# 25岁以下出险次数很低，年轻没钱，买不起车，富二代除外，出险比例比较高，新手技术没到家
# 55岁以上出险次数很低，上岁数了，开车就少了，出险比例比较高，年龄大了，老花眼
# 25-55岁出险比例基本持平
la_cross = pd.crosstab(loss_age.age_label,loss_age.loss_flag)
la_cross.plot(kind='bar')
la_cross.div(la_cross.sum(1),axis=0).plot(kind='bar',stacked=True)
#%%
# 35-45岁之间中年正旺，有钱，车多
stack2dim(loss_age,"age_label","loss_flag")
#%%
# 查看下每个年龄段出险次数所占比例
# 35-45出险次数最多，25-45岁占比高达78.4%
la_cross[1].plot(kind='pie',autopct='%.1f')
#%%
#查看驾龄与出险次数关系
auto_ins_loss.vAge.mean()
#%%
auto_ins_loss.vAge.min()
#%%
auto_ins_loss.vAge.max()
#%%
auto_ins_loss.vAge.median()
#%%
auto_ins_loss.vAge.std()
#%%
auto_ins_loss.vAge.describe()
#%%
auto_ins_loss.vAge.quantile([0.01,0.5,0.99])
#%%
auto_ins_loss.vAge.skew()
#%%
#可以看出驾龄低于2年的出险次数较高，4年以上的出险次数较低
# 老司机开车比较稳了
auto_ins_loss.vAge.hist(bins=10)
#%%
# 中位数在1上，说明新手出险率很高，上四分位数在3，说明老司机都很稳
sns.boxplot(x="loss_flag",y="vAge",data=auto_ins)
#%%
#将驾龄分为这几段[1,2),[2,4)[4,6),[6,8),[8,10.1),进行分析
loss_vage = auto_ins[['vAge','loss_flag']]
bins = pd.IntervalIndex.from_tuples([(1,2),(2,4),(4,6),(6,8),(8,10.1)],closed='left')
vage_label = pd.cut(loss_vage.vAge.values,bins=bins,right=False)
loss_vage['vage_label'] = vage_label
loss_vage.head()
#%%
# 查看每个年龄段的出险次数
# 驾龄一年的出险次数最多，2-3年的次之，新手技术还没到家
# 从出险比例上看，一年的新手出险比例较高，4年以上老司机很稳
lva_cross = pd.crosstab(loss_vage.vage_label,loss_vage.loss_flag)
lva_cross.plot(kind='bar')
lva_cross.div(lva_cross.sum(1),axis=0).plot(kind='bar',stacked=True)
#%%
# 一年级新手次数多，比例高，4年以上老司机很稳
stack2dim(loss_vage,"vage_label","loss_flag")
#%%
# 查看下每个年龄段出险次数所占比例
# 1-3年驾龄所占比例高达80.2%  还是老司机稳
lva_cross[1].plot(kind='pie',autopct='%.1f')
#%%
# 查看汽车出产地和出险的关系
# 出险次数上国产较多，出险比例进口较高，买得起进口车的还是少
loss_ow = pd.crosstab(auto_ins['import'],auto_ins.loss_flag)
loss_ow.plot(kind="bar")
#%%
loss_ow.div(loss_ow.sum(1),axis=0).plot(kind="bar",stacked=True)
#%%
stack2dim(auto_ins,"import","loss_flag")
#%%