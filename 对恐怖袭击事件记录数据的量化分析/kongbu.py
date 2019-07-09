# -*- coding: utf-8 -*-
"""
Created on Sun Sep 16 23:27:55 2018

@author: dell
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Sep 16 10:34:56 2018

@author: lilizhang
"""

import pandas as pd
import numpy as np
import re
from sklearn.cluster import KMeans,DBSCAN
from sklearn.preprocessing import StandardScaler
import collections
from scipy import integrate
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

data=pd.read_excel('D://data/kongbu.xlsx')

event= data.loc[data['eventid'].isin([201701090031, 201702210037, 201703120023, 201705050009, 201705050010, 201707010028, 201707020006, 201708110018, 201711010006, 201712010003])]
#event.to_csv('D:\\data\event.csv',index=None)

data_1=data.drop(['approxdate','resolution','country_txt','region_txt','summary','alternative_txt','attacktype1_txt','attacktype2_txt',
                'attacktype3_txt','targtype1_txt','targsubtype1_txt','natlty1_txt','targtype2_txt','targsubtype2_txt',
                'natlty2_txt','targtype3_txt','targsubtype3_txt','natlty3_txt','claimmode_txt','claimmode2_txt','claimmode3_txt',
                'weaptype1_txt','weapsubtype1_txt','weaptype2_txt','weapsubtype2_txt','weaptype3_txt','weapsubtype3_txt','weaptype4_txt',
                'weapsubtype4_txt','weapdetail','propextent_txt','hostkidoutcome_txt','scite1','scite2','scite3','dbsource','related','provstate','location','gname2','gname3',
                'gsubname2','gsubname3','motive','propcomment','addnotes','ransomnote','corp1','target1','corp2','target2','corp3','target3','city','gsubname','divert','kidhijcountry','nreleased'],axis=1)


gname= data_1['gname']
print(len(gname.value_counts()))
game= pd.Categorical(gname)
#data_1['codes']=game.codes
gamecodes= pd.DataFrame(game.codes)

#data_2= data_1.drop(['gname'],axis=1)
data_2=data_1.fillna(0)
data_2['codes']=game.codes
#print(data_2.columns)

stand= StandardScaler()
data_3= stand.fit_transform(data_2.values)

estimator= KMeans(n_clusters=5)
estimator.fit(data_3)
label= estimator.labels_
#####################################################################################################################################################
'''
第一问
'''
gname_1= collections.Counter(gname)
gname_1=dict(gname_1)
gname_2= list(gname_1)
gname_3= [x for x in gname_1.values()]
gname_2= np.array(gname_2)
gname_3= np.array(gname_3)
gname_4= np.column_stack([gname_2,gname_3])
gname_4= pd.DataFrame(gname_4)
gname_4.rename(columns={0:'gname',1:'count'},inplace=True)
gname_4['count'].replace('59502',0,inplace=True)
gname_4['count']= gname_4['count'].astype('int')

data_4= pd.merge(data_1,gname_4,on=['gname'],how='left')


#将attacktype1从1~9变为8~1，未知为4.5
attacktype1=data_2['attacktype1']
attacktype1=[9-x for x in attacktype1]  
attacktype1=np.array(attacktype1).reshape((len(attacktype1),1))
#attacktype1=pd.DataFrame(attacktype1)
#data_4= pd.concat([data_4,attacktype1],axis=1)

#将weapontype1从1~13变为13~1
weaptype1=data_2['weaptype1']
weaptype1=[14-x for x in weaptype1]
weaptype1=np.array(weaptype1).reshape((len(weaptype1),1))
#weaptype1=pd.DataFrame(weaptype1)
#data_4= pd.concat([data_4,weaptype1],axis=1)

#targtype1  
targtype1=data_2['targtype1']
newX=[20,21,18,19,5,15,17,9,13,6,14,10,3,16,11,7,2,4,10,11.5,8,1]
for i in range(22):
    targtype1=[newX[i] for x in targtype1]
targtype1=np.array(targtype1).reshape((len(targtype1),1))
#targtype1=pd.DataFrame(targtype1)
#data_4= pd.concat([data_4,targtype1],axis=1)

#将propextent从1~4变为4~1
propextent=data_2['propextent']
propextent=[5-x for x in propextent]
propextent=np.array(propextent).reshape((len(propextent),1))
#propextent=pd.DataFrame(propextent)
#data_4= pd.concat([data_4,propextent],axis=1)

#将extened从0~1变为1~2
extended=data_2['extended']
extended=[x+1 for x in extended]
extended=np.array(extended).reshape((len(extended),1))
#extended=pd.DataFrame(extended)
#data_4= pd.concat([data_4,extended],axis=1)

newX= np.column_stack([attacktype1,weaptype1,targtype1,propextent,extended])
newX= pd.DataFrame(newX)


from sklearn.preprocessing import Normalizer
new_data=data_4[['count','nkill','nwound','crit1','crit2','crit3','doubtterr','multiple','attacktype1','weaptype1','propextent','extended','region']]
new_data=pd.concat([new_data,newX],axis=1)
new_data.fillna(0,inplace=True)
norm= Normalizer()
new_data_1= norm.fit_transform(new_data)
#new_data_1=new_data

weightArr=[0.1,10,2,0.1,0.1,0.1,0.1,0.1,0.1,1,2,1,5,1]
new_data_1=np.array(new_data_1).T
result=np.dot(np.array(weightArr),new_data_1)
result=np.array(result).reshape((len(result),1))
result=np.column_stack([data['eventid'],result])
result=result[result[:,1].argsort()] 
#################################################################################

group= data.groupby(['gname'])['eventid', 'gname','latitude','longitude']
print(group)
grouped= dict(list(group))


print(grouped['14 K Triad'])
t=0
for v,k in grouped.items():
    if len(k)>1:
        try:
            k.to_excel('D:\\data\{}.xlsx'.format(v),index=False)
        except:
            t=t+1
            continue
print(t)           
########################################################################################################################################################
'''
第二问
kmeans 聚类
'''
question1= data_2.loc[~data_2['gname'].isin(['Unknown'])]
eventid= np.array(question1['eventid'])
codess= np.array(question1['codes'])
question7= question1.drop(['eventid','gname','codes'],axis=1)

data2015= data_2.loc[data['iyear'].isin([2015,2016])]

question2= data2015.loc[data2015['gname'].isin(['Unknown'])]
eventid_2= np.array(question2['eventid'])
question5= data_2.loc[data_2['eventid'].isin([201701090031, 201702210037, 201703120023, 201705050009, 201705050010, 201707010028, 201707020006, 201708110018, 201711010006, 201712010003])]
question2= question2.drop(['gname','codes','eventid'],axis=1)
question5= question5.drop(['gname','codes','eventid'],axis=1)
#dbscan= DBSCAN(eps=0.8,min_samples=5,metric='euclidean',algorithm='auto')
question3= question7[['country','region','latitude','longitude','attacktype1','targtype1','natlty1','suicide','nperps','claimed','claimmode','weaptype1','weapsubtype1','extended','nkillus','ransom']]
question4= question2[['country','region','latitude','longitude','attacktype1','targtype1','natlty1','suicide','nperps','claimed','claimmode','weaptype1','weapsubtype1','extended','nkillus','ransom']]
question6= question5[['country','region','latitude','longitude','attacktype1','targtype1','natlty1','suicide','nperps','claimed','claimmode','weaptype1','weapsubtype1','extended','nkillus','ransom']]

#stad=StandardScaler()
#stad.fit_transform(question3)
#dbscan.fit(question3)
#y_pred= pd.DataFrame(dbscan.labels_)

kmeans= KMeans(n_clusters=700)
kmeans.fit(question3)
y_pred_1= kmeans.labels_
grouppred= np.column_stack([eventid,codess,y_pred_1])
grouppred= pd.DataFrame(grouppred)
pivot= grouppred.groupby([2])
pivot_1= dict(list(pivot))


pred= kmeans.predict(question4.values)
pred_results= np.column_stack([eventid_2,pred])
pred_results= pd.DataFrame(pred_results)
pred_pivot= pred_results.groupby([1])
pred_pivot_1= dict(list(pred_pivot))

pred_results_= pred_results.loc[pred_results[0].isin([201701090031, 201702210037, 201703120023, 201705050009, 201705050010, 201707010028, 201707020006, 201708110018, 201711010006, 201712010003])]
#########################################################################################################################################################################################################
'''
计算危害性 并排序
'''
table= pd.read_csv('D://data/result1.csv')
allsum=[]
i=0
table= table.rename(columns={'0':'eventid'})
table['eventid']= data['eventid']
table['eventid']= table['eventid'].astype('int64')

for value in pivot_1.values():
    value= value.rename(columns={0:'eventid',1:'group'})
    value['eventid']= value['eventid'].astype('int64')
    value= pd.merge(table,value,on='eventid',how='right')
    nums= value['1'].mean()
    allsum.append((i,nums))
    i=i+1
allsum_1= allsum.sort(key=lambda x:x[1]) 

#####################################################################################################################################################################
'''
计算相关性
'''
def calEuclideanDistance(vec1,vec2):  
    dist = np.sqrt(np.sum(np.square(vec1 - vec2)))
    return dist

dis=[]
question6=np.array(question6)
for i in range(len(question6)):
    test= question6[i,:]
    a=120
    mindis=999999
    for j in [120,87,454,463,291]:
        value_3= pivot_1[j]
        value_3=np.array(value_3)
        value_4= question1.loc[question1['eventid'].isin(value_3[:,0])]
        value_4= value_4[['country','region','latitude','longitude','attacktype1','targtype1','natlty1','suicide','nperps','claimed','claimmode','weaptype1','weapsubtype1','extended','nkillus','ransom']]
        value_4= np.array(value_4)
        distances=0
        for k in range(len(value_4)):
            train= value_4[k,:]
            distance= calEuclideanDistance(test,train)
            distances= distances + distance
        distances=distances/len(value_4)

        dis.append(distances)
#################################################################################################################################################################

for j in [120,87,454,463,291]:
    valuen=pivot_1[j]
    valuen=np.array(valuen)
    value_= question1.loc[question1['eventid'].isin(valuen[:,0])]
    value_= value_[['country','region','latitude','longitude','attacktype1','targtype1','natlty1','suicide','nperps','claimed','claimmode','weaptype1','weapsubtype1','extended','nkillus','ransom']]
    value_.to_csv('D://group/group1/{}.csv'.format(str(j)))
#####################################################################################################################################################################

'''
空间特性
计算袭击目标jini指数
'''
def gini(mx):
    area=0
    for i in range(len(mx)-1):
        aa= ((mx[i,1]+ mx[i+1,1])*(mx[i+1,0]-mx[i,0]))/2
        area+=aa
    return 1-2*area
    

data_2015= data.loc[data['iyear'].isin([2015,2016,2017])]
targettype= data_2015.groupby(['targtype1']).size().sort_values()
print(targettype)
tar_sum=targettype.sum()
tar_pin= targettype/tar_sum
print(tar_pin)
tar_cumsum= tar_pin.cumsum()
print(tar_cumsum)

target_m= np.linspace(0,1,23,endpoint=True)
target_m= np.delete(target_m,0,0)
target_x = np.array(list(tar_cumsum))
target_mx= np.column_stack([target_m,target_x])

target_mx_= pd.DataFrame(target_mx)
target_mx_.to_csv('D://group/target.csv',header=False,index=False)

target_gini= gini(target_mx)
print(target_gini)

'''
计算国家jini指数
'''
print(data_2015.shape[0])
print(data_2015['nkill'].sum())
print(data_2015['nwound'].sum())
print(len(data_2015['country'].value_counts()))
country= data_2015.groupby(['country']).size().sort_values()
print(country)
con_sum=country.sum()
con_pin= country/con_sum
print(con_pin)
con_cumsum= con_pin.cumsum()
print(con_cumsum)

con_m= np.linspace(0,1,133,endpoint=True)
con_m= np.delete(con_m,0,0)
con_x = np.array(list(con_cumsum))

con_mx= np.column_stack([con_m,con_x])

con_mx_= pd.DataFrame(con_mx)
con_mx_.to_csv('D://group/country.csv',header=False,index=False)

con_gini= gini(con_mx)
print(con_gini)

'''
计算武器基尼指数
'''
weaptype= data_2015.groupby(['weaptype1']).size().sort_values()
print(weaptype)
weap_sum= weaptype.sum()
weap_pin= weaptype/weap_sum
print(weap_pin)
weap_cumsum= weap_pin.cumsum()
print(weap_cumsum)

weap_m= np.linspace(0,1,11,endpoint=True)
weap_m= np.delete(weap_m,0,0)
weap_x = np.array(list(weap_cumsum))
weap_mx= np.column_stack([weap_m,weap_x])

weap_mx_= pd.DataFrame(weap_mx)
weap_mx_.to_csv('D://group/weapon.csv',header=False,index=False)

weap_gini= gini(weap_mx)
print(weap_gini)

'''
计算攻击类型基尼指数
'''
attacktype= data_2015.groupby(['attacktype1']).size().sort_values()
print(attacktype)
attack_sum= attacktype.sum()
attack_pin= attacktype/attack_sum
print(attack_pin)
attack_cumsum= attack_pin.cumsum()
print(attack_cumsum)

attack_m= np.linspace(0,1,10,endpoint=True)
attack_m= np.delete(attack_m,0,0)
attack_x = np.array(list(attack_cumsum))
attack_mx= np.column_stack([attack_m,attack_x])

attack_mx_= pd.DataFrame(attack_mx)
attack_mx_.to_csv('D://group/attacktype.csv',header=False,index=False)

attack_gini= gini(attack_mx)
print(attack_gini)

'''
计算region 基尼指数
'''
region= data_2015.groupby(['region']).size().sort_values()
print(region)
region_sum= region.sum()
region_pin= region/region_sum
print(region_pin)
region_cumsum= region_pin.cumsum()
print(region_cumsum)

region_m= np.linspace(0,1,13,endpoint=True)
region_m= np.delete(region_m,0,0)
region_x = np.array(list(region_cumsum))
region_mx= np.column_stack([region_m,region_x])

region_mx_= pd.DataFrame(region_mx)
region_mx_.to_csv('D://group/region.csv',header=False,index=False)

region_gini= gini(region_mx)
print(region_gini)
#########################################################################################################################################################################
'''
国家层面
'''
country= data_2015.groupby(['country']).size().sort_values()
print(country)
country_= list(country)[::-1]
plt.figure(figsize=(8,10))
plt.plot(country_)
plt.show()

country_=pd.DataFrame(country_)
country_.to_csv('D://group/pinlv.csv',header=None)
'''
武器
'''
weap_con= data_2015.groupby(['weaptype1'])

print(weap_con.size())

i=0
for key,group in weap_con:
    group1= group.groupby(['country']).size().sort_values()
    print(group1)
    group1_sum= group1.sum()
    group1_pin= group1/group1_sum
    #print(group1_pin)
    group1_cumsum= group1_pin.cumsum()
    #print(group1_cumsum)
    
    group1_m= np.linspace(0,1,len(group1)+1,endpoint=True)
    group1_m= np.delete(group1_m,0,0)
    group1_x = np.array(list(group1_cumsum))
    group1_mx= np.column_stack([group1_m,group1_x])
    '''
    print(1-group1_x[-int(0.05*len(group1))-1])
    print(1-group1_x[-int(0.1*len(group1))-1])
    print(1-group1_x[-int(0.2*len(group1))-1])
    '''
    group1_mx_= pd.DataFrame(group1_mx)
    group1_mx_.to_csv('D://group/group_weaptype{}.csv'.format(i),header=False,index=False)
    group1_gini= gini(group1_mx)
    print(key,group1_gini)
    i=i+1

'''
袭击方式
'''
attack_con= data_2015.groupby(['attacktype1'])

print(attack_con.size())

i=0
for key,group in attack_con:
    group1= group.groupby(['country']).size().sort_values()
    #print(group1)
    group1_sum= group1.sum()
    group1_pin= group1/group1_sum
    #print(group1_pin)
    group1_cumsum= group1_pin.cumsum()
    #print(group1_cumsum)
    
    group1_m= np.linspace(0,1,len(group1)+1,endpoint=True)
    group1_m= np.delete(group1_m,0,0)
    group1_x = np.array(list(group1_cumsum))
    group1_mx= np.column_stack([group1_m,group1_x])
    print(1-group1_x[-int(0.05*len(group1))-1])
    print(1-group1_x[-int(0.1*len(group1))-1])
    print(1-group1_x[-int(0.2*len(group1))-1])
    
    
    group1_mx_= pd.DataFrame(group1_mx)
    group1_mx_.to_csv('D://group/group_attacktype{}.csv'.format(i),header=False,index=False)
    group1_gini= gini(group1_mx)
    print(key,group1_gini)
    i=i+1

'''
袭击目标
'''
target_con= data_2015.groupby(['targtype1'])

print(target_con.size())
q1=[]
q2=[]
q3=[]
i=0
for key,group in target_con:
    group1= group.groupby(['country']).size().sort_values()
    #print(group1)
    group1_sum= group1.sum()
    group1_pin= group1/group1_sum
    #print(group1_pin)
    group1_cumsum= group1_pin.cumsum()
    #print(group1_cumsum)
    
    group1_m= np.linspace(0,1,len(group1)+1,endpoint=True)
    group1_m= np.delete(group1_m,0,0)
    group1_x = np.array(list(group1_cumsum))
    group1_mx= np.column_stack([group1_m,group1_x])
    q1.append(1-group1_x[-int(0.05*len(group1))-1])
    q2.append(1-group1_x[-int(0.1*len(group1))-1])
    q3.append(1-group1_x[-int(0.2*len(group1))-1])
    
    
    group1_mx_= pd.DataFrame(group1_mx)
    group1_mx_.to_csv('D://group/group_targettype{}.csv'.format(i),header=False,index=False)
    group1_gini= gini(group1_mx)
    print(key,group1_gini)
    i=i+1

###############################################################################################################################################################
'''
时间特性
频率特性
'''
year= data_2015.groupby(['iyear'])

print(year.size())

i=0
for key,group in year:
    group1= group.groupby(['imonth'])
    for key1,group2 in group1:
        group3= group2.groupby(['iday']).size().sort_values()
        #print(group1)
        group3_sum= group3.sum()
        group3_pin= group3/group3_sum
        #print(group1_pin)
        group3_cumsum= group3_pin.cumsum()
        #print(group1_cumsum)
    
        group3_m= np.linspace(0,1,len(group3)+1,endpoint=True)
        group3_m= np.delete(group3_m,0,0)
        group3_x = np.array(list(group3_cumsum))
        group3_mx= np.column_stack([group3_m,group3_x])

        #group1_mx_= pd.DataFrame(group1_mx)
        #group1_mx_.to_csv('D://group/group_weaptype{}.csv'.format(i),header=False,index=False)
        group3_gini= gini(group3_mx)
        print(key1,group3_gini)
        i=i+1

i=0
for key,group in year:
    group1= group.groupby(['imonth']).size().sort_values()
        
    #print(group1)
    group1_sum= group1.sum()
    group1_pin= group1/group1_sum
    #print(group1_pin)
    group1_cumsum= group1_pin.cumsum()
    #print(group1_cumsum)
    
    group1_m= np.linspace(0,1,len(group1)+1,endpoint=True)
    group1_m= np.delete(group1_m,0,0)
    group1_x = np.array(list(group1_cumsum))
    group1_mx= np.column_stack([group1_m,group1_x])

    #group1_mx_= pd.DataFrame(group1_mx)
    #group1_mx_.to_csv('D://group/group_weaptype{}.csv'.format(i),header=False,index=False)
    group1_gini= gini(group1_mx)
    print(key,group1_gini)
    i=i+1

'''
受袭击国家数目和时间特性
'''
attack_country= data_2015.groupby(['iyear'])

for key, group in attack_country:
    group1=group['nkill'].groupby(group['imonth']).sum().sort_values()
    print(group1)
    group1_sum= group1.sum()
    group1_pin= group1/group1_sum
    #print(group1_pin)
    group1_cumsum= group1_pin.cumsum()
    #print(group1_cumsum)
    
    group1_m= np.linspace(0,1,len(group1)+1,endpoint=True)
    group1_m= np.delete(group1_m,0,0)
    group1_x = np.array(list(group1_cumsum))
    group1_mx= np.column_stack([group1_m,group1_x])

    #group1_mx_= pd.DataFrame(group1_mx)
    #group1_mx_.to_csv('D://group/group_weaptype{}.csv'.format(i),header=False,index=False)
    group1_gini= gini(group1_mx)
    print(key,group1_gini)

##############################################################################################################################################################
'''
因子分析
'''
from sklearn.preprocessing import StandardScaler

data_1995= data.loc[data['iyear'].isin([2015,2016])]
data_1997= data.loc[data['iyear'].isin([2017])]
dead= data_1995['nkill'].groupby(data_1995['country']).sum().sort_values()
dead_cont= data_1995.groupby(data_1995['country']).size().sort_values()

def qushi(datas):
    
    time_country= datas.groupby(['country']).size() #总起数
    time_con= dict(time_country)

    dead= datas['nkill'].groupby(datas['country']).sum()
    dead_con= dict(dead)

    wound= datas['nwound'].groupby(datas['country']).sum()
    wound_con= dict(wound)

    propextent_=datas['propextent'].fillna(5)
    propextent_=[5-x for x in propextent_]
    propextent_=pd.DataFrame(np.array(propextent_))
    data_2016= datas.reset_index(drop=True)
    data_2016= pd.concat([data_2016,propextent_],axis=1)

    poperty= data_2016[0].groupby(data_2016['country']).sum()
    poperty_con= dict(poperty)

    time_dead= [1 if datas['nkill'].iloc[i]>=10 else 0 for i in range(datas.shape[0])]
    time_dead= pd.DataFrame(np.array(time_dead),columns=['time_dead'])
    data_2017= pd.concat([data_2016,time_dead],axis=1)
    time_death= data_2017['time_dead'].groupby(data_2017['country']).sum()
    time_death_con= dict(time_death)

    yinzi=[]
    k=[]
    for key in time_con.keys():
        z=[]
        k.append(key)
        z.append(time_death_con[key])
        z.append(time_con[key])
        z.append(dead_con[key])
        z.append(wound_con[key])
        z.append(poperty_con[key])
        yinzi.append(z)

    yinzi= np.array(yinzi).reshape(-1,5)
    yinzi= pd.DataFrame(yinzi)
    yinzi=yinzi.fillna(0)

    percent= yinzi[2]/yinzi[1]
    dead_percent= yinzi[0]/yinzi[1]

    percent= pd.DataFrame(percent)
    dead_percent= pd.DataFrame(dead_percent)
    yinzi= pd.concat([yinzi,percent,dead_percent],axis=1)


    stand= StandardScaler()
    yinzi_= stand.fit_transform(yinzi)

    pca = PCA(n_components=2)
    pca.fit(yinzi_)
    print(pca.n_components_)
    variance1= pca.explained_variance_ratio_
    #variance2= pca.explained_variance_
    #variannce3= pca.explained_variance_
    conponents= pca.components_ 
    return variance1,conponents,yinzi_,k
#print(pca.singular_values_)
variance1,conponent1,yinzi1,k1= qushi(data_1995)
variance2,conponent2,yinzi2,k2= qushi(data_1997)


weight1= [0.444,0.440,0.450,0.445,0.432,0.116,0.095]
weight2= [-0.008,-0.114,-0.031,-0.074,-0.104,0.692,0.700]

weight1= np.array(weight1)
weight2= np.array(weight2)

F1= np.dot(yinzi1,weight1)
F2= np.dot(yinzi1,weight2)

F3= 0.685*F1+ 0.264*F2
k1=np.array(k1)
All= np.column_stack([k1,F3])
All= np.column_stack([All,F1])
All= np.column_stack([All,F2])
All= pd.DataFrame(All)
All_1=All.sort_values(by=[1],ascending=False)

weight3= [0.443,0.435,0.451,0.445,0.404,0.163,0.152]
weight4= [-0.027,-0.166,-0.045,-0.094,-0.170,0.679,0.686]

weight3= np.array(weight3)
weight4= np.array(weight4)

F4= np.dot(yinzi2,weight3)
F5= np.dot(yinzi2,weight4)

F6= 0.672*F4+ 0.261*F5
k2=np.array(k2)
All1= np.column_stack([k2,F6])
All1= np.column_stack([All1,F4])
All1= np.column_stack([All1,F5])
All1= pd.DataFrame(All1)
All_2=All1.sort_values(by=[1],ascending=False)
All_2=All_2.reset_index(drop=True)

paiming= [i for i in range(All_2.shape[0])]
paiming= pd.DataFrame(paiming)
All_2= pd.concat([All_2,paiming],axis=1)


All_1.to_excel('D://group/all_1.xlsx',index=None,header=None)
All_2.to_excel('D://group/all_2.xlsx',index=None,header=None)

