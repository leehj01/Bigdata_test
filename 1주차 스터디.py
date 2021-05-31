# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.7.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import pandas as pd
import numpy as np

# ### 타이타닉 데이터

train = pd.read_csv('titanic/train.csv')

# 1. titanic 데이터의 Embarked 컬럼에서 결측값에 대하여 Embarked 컬럼의 최빈값으로 대체하시오.

train['Embarked'].value_counts().idxmax()

# +
print('Embarked 는 카테고리 변수',set(train['Embarked']))
mode_em = train['Embarked'].value_counts().idxmax()
print('최빈값 : ', mode_em)

train['Embarked'].fillna(mode_em, inplace= True)
train['Embarked'] = train['Embarked'].fillna(mode_em)

print('Embarked 는 카테고리 변수',set(train['Embarked']))
# -

# 2. titanic 데이터의 ‘Fare’ 컬럼에서 1.5*IQR 범위를 벗어나는 값의 개수를 구하시오.
#

# +
q1 , q3 = np.quantile(train.Fare , 0.25) , np.quantile(train.Fare , 0.75)
IQR = q3 - q1 
out = IQR * 1.5 
lower , upper = q1 - out , q3 + out

# 아래로 벗어나는 값과, 위로 벗어나는 값 
data1 , data2 = train[train['Fare'] < lower ] ,train[train['Fare'] > upper ] 

print('이상치 데이터 합 : ' , data1.shape[0] + data2.shape[0])
# -

# ### 와인 데이터

wine_train = pd.read_csv('wine/train.csv')

wine_train.head()

# 문제 1. winequality-red.csv 데이터에서 결측치가 있는 컬럼이 존재하는지 확인하기. 
#

wine_train.isnull().sum()

# 문제 2.   winequality-red.csv 에서 quality 가 5인 데이터의 알콜농도의 평균을 구하여라

wine_train[wine_train['quality'] == 5]['alcohol'].mean()

wine_train.loc[wine_train['quality']==5,'alcohol'].mean()

# ### world-happiness-report 

report_train = pd.read_csv('world-happiness-report.csv')
report_train.head()

# 1. world-happiness-report.csv 데이터에서 결측치가 있는 컬럼의 값들을 "0"으로 변환하기

report_train.isnull().sum()

report_train.columns

report_train[(report_train['Generosity'].isnull()) | (report_train['Negative affect'].isnull()) ].index

report_train = report_train.fillna(0)

report_train.isnull().sum().sum()

# 2. country name 컬럼에서 south korea 데이터만 추출하기

report_train[report_train['Country name'] =='South Korea'].head()

report_train.loc[report_train['Country name'] =='South Korea',:].head()

train = pd.read_csv('movie/movies_train.csv')
train.to_csv('movie/movies_train.csv', encoding = 'utf-8-sig', index= False)

train.to_csv('movie/movies_train.csv', encoding = 'utf-8-sig', index= False)


