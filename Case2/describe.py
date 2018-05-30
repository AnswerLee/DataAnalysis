# -*- coding: utf-8 -*-
"""
Created on Wed May 30 19:48:46 2018

@author: AnswerLee
"""

#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

#%%
os.chdir(r'D:\give_me_five\githome\DataAnalysis\Case2')
snd = pd.read_csv('./dataset/sndHsPr.csv')
snd.head()
#%%
snd["total_price"] = snd[["AREA","price"]].apply(lambda x:x[0]*x[1],axis=1)
snd.head()
#%%
#先作频次统计，然后绘制柱形图图展现每个区样本的数量,将拼音使用汉字来表示
dist_names = {"chaoyang":"朝阳区","dongcheng":"东城区",
              "fengtai":"丰台区","haidian":"海淀区",
              "shijingshan":"石景山区","xicheng":"西城区"}
snd["dist_name"] = snd.dist.map(dist_names)
snd.head()
#%%
#单因子频数:描述名义变量的分布
#如果遇到中文乱码，则使用如下代码
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei'] # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题
snd.dist_name.value_counts().plot(kind="bar")
#%%
snd.dist_name.value_counts().plot(kind="pie")
#%%
#描述连续变量的分布
snd.price.mean()
#%%
snd.price.max()
#%%
snd.price.min()
#%%
snd.price.std()
#%%
snd.price.median()
#%%
#查看单价的最大、最小值、标准差、中位数，均值，
snd.price.describe()
#%%
snd.price.quantile([0.01,0.5,0.99])
#%%
#偏度值大于零，右偏
snd.price.skew()
#%%
#柱状图查看单价的分布情况
snd.price.hist(bins=40)
#%%
#表分析
#两个名义变量的分布 stacked表示堆叠
ctab1 = pd.crosstab(snd.dist_name,snd.school)
ctab1.plot(kind="bar",stacked=True)
type(ctab1)
#%%
ctab2 = pd.crosstab(snd.dist_name,snd.subway)
ctab2.plot(kind="bar",legend="best")
#%%
sub_sh = pd.crosstab(snd.dist_name,snd.school)
sub_sh["sum1"] = sub_sh.sum(1)
#计算出比例值,画出比例图
sub_sh = sub_sh.div(sub_sh.sum1,axis=0)
sub_sh[[0,1]].plot(kind="bar",stacked=True)
#%%
#画两个维度的堆叠柱状图
#柱状图宽度和总数的大小成正比
from stack2dim import *
stack2dim(snd,"dist_name","school")
#%%
#分类变量和连续变量之间的关系
#箱式图 盒须图
sns.boxplot(x="dist_name",y="price",data=snd)
#%%
#分类变量和连续变量之间的关系
#均值
snd.groupby(["dist_name"])["price"].mean().plot(kind="bar")
#%%
snd.groupby(["dist_name"])["price"].mean().sort_values(ascending=True).plot(kind="barh")
#%%
from pyecharts import Map
#from echarts-china-cities-pypkg import *
"""
官网给的解释如下：

自从 0.3.2 开始，为了缩减项目本身的体积以及维持 pyecharts 项目的轻量化运行，pyecharts 将不再自带地图 js 文件。如用户需要用到地图图表，可自行安装对应的地图文件包。下面介绍如何安装。

全球国家地图: echarts-countries-pypkg (1.9MB): 世界地图和 213 个国家，包括中国地图
中国省级地图: echarts-china-provinces-pypkg (730KB)：23 个省，5 个自治区
中国市级地图: echarts-china-cities-pypkg (3.8MB)：370 个中国城市:https://github.com/echarts-maps/echarts-china-cities-js
pip install echarts-countries-pypkg
pip install echarts-china-provinces-pypkg
pip install echarts-china-cities-pypkg
别注明，中国地图在 echarts-countries-pypkg 里。
"""
snd_price = list(zip(snd.price.groupby(snd.dist_name).mean().index,
                  snd.price.groupby(snd.dist_name).mean().values))
attr, value = Map.cast(snd_price)
min_ = snd.price.groupby(snd.dist).mean().min()
max_ = snd.price.groupby(snd.dist).mean().max()

price_map = Map('北京各区房价', width = 1200, height = 600)
price_map.add('', attr, value, maptype = '北京', is_visualmap = True, visual_range=[min_, max_], 
        visual_text_color = '#000', is_label_show =True)
price_map.render()
#生成render.html在当前目录
#%%
#汇总表
pd.pivot_table(snd,values="price",index="dist_name",columns="school",aggfunc=np.mean).plot(kind="bar")
#%%
#两个连续变量做散点图，查看面积和单价的关系
snd.plot.scatter(x="AREA",y="price")
#%%
#双轴图  导入GDP数据
gdp_gdpcr = pd.read_csv("./dataset/gdp_gdpcr.csv",encoding='gbk')
gdp_gdpcr.head()
#%%
fig = plt.figure()

ax1 = fig.add_subplot(111)
ax1.bar(gdp_gdpcr.year,gdp_gdpcr.GDP)
ax1.set_ylabel("GDP")
ax1.set_title("GDP of China(2000-2017)")
ax1.set_xlim(2000,2017)


ax2 = ax1.twinx()
ax2.plot(gdp_gdpcr.year,gdp_gdpcr.GDPCR,'r')
ax2.set_ylabel("Increase Ratio")
ax2.set_xlabel("Year")
#%%

