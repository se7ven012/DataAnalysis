#%%
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#%%
train_data = pd.read_csv('titanic_train.csv')
test_data = pd.read_csv('titanic_test.csv')

sns.set_style('whitegrid')
#看看头几行
train_data.head()

# %%
#看看数据
train_data.info()
print("-" * 40)
test_data.info()

# %%
#存活：1，去世：0
#原始数据中性别是string 这里处理一下 因为sklearn的模型只能处理数值属性
train_data.loc[train_data["Sex"] == "male","Sex"] = 0
train_data.loc[train_data["Sex"] == "female","Sex"] = 1
#年龄可能是一个比较重要的数据，这里我们使用中位数填充
train_data["Age"] = train_data['Age'].fillna(train_data['Age'].median())  
#Embarked这一属性(在哪儿上船)有缺失，我们用众数赋值，然后把string替换成int
train_data.Embarked[train_data.Embarked.isnull()] = train_data.Embarked.dropna().mode().values
train_data.loc[train_data["Embarked"] == "S","Embarked"] = 0    
train_data.loc[train_data["Embarked"] == "C","Embarked"] = 1
train_data.loc[train_data["Embarked"] == "Q","Embarked"] = 2
#Cabin这一属性（仓位）有缺失，也可能代表没有仓位，这里标记成‘U0’
train_data['Cabin'] = train_data.Cabin.fillna('U0')    
#train_data.Cabin[train_data.CAbin.isnull()]='U0'

#再看一下数据
train_data.info()
#%%
#这里分析一下数据关系

#性别与生存的关系
train_data[['Sex','Survived']].groupby(['Sex']).mean()
train_data[['Sex','Survived']].groupby(['Sex']).mean().plot.bar()

#结果显示的确是男的死得多Q^Q

#%%
#船舱等级与生存的关系
#train_data[['Pclass','Survived']].groupby(['Pclass']).mean()
#train_data[['Pclass','Survived']].groupby(['Pclass']).mean().plot.bar()

#结果显示富裕阶层活下来的更多Q…Q

#不同船舱等级男女存活率
train_data[['Sex','Pclass','Survived']].groupby(['Pclass','Sex']).mean()
train_data[['Sex','Pclass','Survived']].groupby(['Pclass','Sex']).mean().plot.bar()
#%%
#年龄与生存的关系
fig,ax = plt.subplots(1,2, figsize = (18,5))
ax[0].set_yticks(range(0,110,10))
sns.violinplot("Pclass","Age",hue="Survived",data=train_data,split=True,ax=ax[0])
ax[0].set_title('Pclass and Age vs Survived') 

ax[1].set_yticks(range(0,110,10))
sns.violinplot("Sex","Age",hue="Survived",data=train_data,split=True,ax=ax[1])
ax[1].set_title('Sex and Age vs Survived')
 
plt.show()

#%%
#不同年龄段下的平均生存率
fig,axis1 = plt.subplots(1,1,figsize=(18,4))
train_data['Age_int'] = train_data['Age'].astype(int)
average_age = train_data[["Age_int", "Survived"]].groupby(['Age_int'],as_index=False).mean()
sns.barplot(x='Age_int',y='Survived',data=average_age)

print(train_data['Age'].describe())
#这里有891个样本，平均年龄31岁，标准差13.5岁，最小/大年龄分别为0.42/80
#按照年龄组将他们划分成 儿童，少年，成年，老年 分析一下群体生存几率
ages = [0, 12, 18, 65, 100]
train_data['Age_group'] = pd.cut(train_data['Age'],ages)
by_age = train_data.groupby('Age_group')['Survived'].mean()
by_age.plot(kind = 'bar')
print(by_age)

#%%
#人物名称/头衔与存活率
train_data['Title'] = train_data['Name'].str.extract(' ([A-Za-z]+)\.',expand=False)
#打印一下头衔
pd.crosstab(train_data['Title'],train_data['Sex'])
train_data[['Title','Survived']].groupby(['Title']).mean().plot.bar()

#%%
#名字长度与存活率。。。
fig, axis1 = plt.subplots(1,1,figsize=(18,4))
train_data['Name_length'] = train_data['Name'].apply(len)
name_length = train_data[['Name_length','Survived']].groupby(['Name_length'], as_index=False).mean()
sns.barplot(x='Name_length', y='Survived',data=name_length)

#%%
#票价与存活率
fare_not_survived = train_data['Fare'][train_data['Survived'] == 0]
fare_survived = train_data['Fare'][train_data['Survived'] == 1]
 
average_fare = pd.DataFrame([fare_not_survived.mean(),fare_survived.mean()])
std_fare = pd.DataFrame([fare_not_survived.std(),fare_survived.std()])
average_fare.plot(yerr=std_fare,kind='bar',legend=False)
 
plt.show()
#结果显示还是有一定相关性的 存活者的票价均值大于死者

#%%
###特征工程太复杂了没有做


#%%

#线性回归
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import KFold

#predictors =["Pclass","Age","SibSp","Parch","Fare"]
predictors = ["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"]

#initialize
alg=LinearRegression()
#样本均分成3份进行交叉验证
#kf = KFold(data_train.shape[0],n_folds=3,random_state=1)   
kf = KFold(n_splits=3,shuffle=False,random_state=1)

predictions = []
for train,test in kf.split(train_data):
    train_predictors = (train_data[predictors].iloc[train,:])
    train_target = train_data["Survived"].iloc[train]

    alg.fit(train_predictors,train_target)
    
    test_predictions = alg.predict(train_data[predictors].iloc[test,:])
    predictions.append(test_predictions)

#把三份样本粘回axis0上
predictions = np.concatenate(predictions,axis=0)
#筛选一下结果 .5以上算活着 反之则算去世
predictions[predictions>.5] = 1
predictions[predictions<=.5] = 0
accuracy = sum(predictions==train_data["Survived"])/len(predictions)
print("Accuracy: ",accuracy)
#0.7037
#0.7834

# %%

#逻辑回归
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression

predictors =["Pclass","Age","SibSp","Parch","Fare"]
#predictors = ["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"]

#initialize
LogRegAlg=LogisticRegression(random_state=1)
re = LogRegAlg.fit(train_data[predictors],train_data["Survived"])

#使用交叉验证函数获取预测准确率分数
scores = model_selection.cross_val_score(LogRegAlg,train_data[predictors],train_data["Survived"],cv=3)
#取平均值作为最终准确率
print("Accuracy: ",scores.mean())
#0.6992
#0.7878

# %%

#随机森林
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier

#predictors =["Pclass","Age","SibSp","Parch","Fare"]
predictors = ["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"]

#10棵决策树，停止的条件：样本个数为2，叶子节点个数为1
#alg=RandomForestClassifier(random_state=1,n_estimators=10,min_samples_split=2,min_samples_leaf=1) 

#30棵决策树，停止的条件：样本个数为2，叶子节点个数为1
alg=RandomForestClassifier(random_state=1,n_estimators=30,min_samples_split=2,min_samples_leaf=1) 

#样本均分成3份进行交叉验证
#kf = KFold(data_train.shape[0],n_folds=3,random_state=1)   
kf = KFold(n_splits=3,shuffle=False, random_state=1)

scores = model_selection.cross_val_score(alg,train_data[predictors],train_data["Survived"],cv=kf)
#取平均值作为最终准确率
print("Accuracy: ",scores.mean())
#0.652
#0.785

#0.658
#0.796


# %%
