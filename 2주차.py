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

train = pd.read_csv('house/train.csv')
test = pd.read_csv('house/test.csv')

#  ### 문제 1
#
#  1. 2021 에서, YrSold 년도를 뺀 연도를 'CharYr' 으로 만들어라. 단, 2010년은 그대로 2010년을 유지해라

# 이 두개 라이브러리는 시험범위가 아님으로 무시하셔도 됩니다.
import copy
import time

t = copy.deepcopy(train)  # 깊은 복사를 위한 코드
# 현재시간과 연도 구하는 코드
now = time.time()        
now_year = time.strftime('%Y', time.localtime(time.time()))

set(t['YrSold'])  # YrSold 는 2006, 2007, 2008, 2009, 2010 을 가짐

t.columns

# #### 문제 푸는 방법 1. - map/lambda 사용하기
#

t['CharYr'] = t['YrSold'].map(lambda x : int(now_year) - x if x != 2010 else x )


# #### 문제 푸는 방법 2. 함수와 map 사용

def char(x):
    if x != 2010 :
        return 2021 - x 
    else :
        return x 


t['CharYr'] = t['YrSold'].map(char )

# #### 문제 결과

t[['YrSold','CharYr']]

# 2. SaleType 에 대한 중복값을 확인하고, SaleType 의 중복값을 제거해라

t[t.duplicated('MSZoning')]

t.drop_duplicates('SaleType', inplace = True)
