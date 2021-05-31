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

# ### Data 및 문제 

# - ① y_train.csv : 고객의 성별 데이터 (학습용), CSV 형식의 파일
# - ② X_train.csv, X_test.csv : 고객의 상품구매 속성 (학습용 및 평가용), CSV 형식의 파일
#
# 고객 3,500명에 대한 학습용 데이터(y_train.csv, X_train.csv)를 이용하여 성별예측 모형을 만든 
# 후, 이를 평가용 데이터(X_test.csv)에 적용하여 얻은 2,482명 고객의 성별 예측값(남자일 확률)을 
# 다음과 같은 형식의 CSV 파일로 생성하시오.(제출한 모델의 성능은 ROC-AUC 평가지표에 따라 
# 채점) 
#
# - 0  : 여자, 1  남자 
#

# +
import pandas as pd
import matplotlib.pyplot as plt
from  matplotlib import font_manager, rc
import seaborn as sns
import platform

if platform.system() == 'Windows':
    font_name = font_manager.FontProperties(fname = 'c:/Windows/Fonts/malgun.ttf').get_name()
    rc('font', family = font_name)
else:
    pass

import warnings
warnings.filterwarnings(action ='ignore')
# -

x_train = pd.read_csv('X_train.csv', encoding = 'euc_kr')
y_train = pd.read_csv('y_train.csv', encoding = 'euc_kr')
x_test = pd.read_csv('X_test.csv', encoding = 'euc_kr')
x_train

y_train

# data 를 합해서 하나의 train 데이터로 만들기 
train = x_train.merge(y_train, how = 'inner', left_on = 'cust_id', right_on = 'cust_id')
train

# # 데이터 EDA 

train.info()
# 환불 금액만 NULL 이 있는 것을 확인 할 수 있다. 

train.describe(include = 'all')

# ## 남자 / 여자에 대해서 어떻게 다른지 파악하기

man_train = train[train['gender'] == 1 ]
woman_train = train[train['gender'] == 0]

### 주 구매지 파악 
plt.figure(figsize = (15,8))
sns.countplot( data = train, x = '주구매지점'  , hue = 'gender')
plt.xticks( rotation = 70 )
plt.show()


