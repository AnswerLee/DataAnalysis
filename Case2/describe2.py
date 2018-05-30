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
auto_ins = pd.read_csv("./dataset/auto_ins.csv",encoding='gbk')
auto_ins.head()
#%%
auto_ins["loss_flag"] = auto_ins.Loss.map(lambda x:x if x == 0 else 1).astype('Int64')
auto_ins.head()
#%%
#对loss_flag单分类变量进行分析  频数
auto_ins.loss_flag.value_counts().plot(kind="bar")
#%%
#饼状图
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
# 查看汽车出产地和出险的关系
# 出险次数上国产较多，出险比例进口较高，买得起进口车的还是少
loss_ow = pd.crosstab(auto_ins['import'],auto_ins.loss_flag)
loss_ow.plot(kind="bar")
#%%
loss_ow.div(loss_ow.sum(1),axis=0).plot(kind="bar",stacked=True)
#%%
stack2dim(auto_ins,"import","loss_flag")
#%%