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

# +
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler, robust_scale, RobustScaler

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings(action='ignore')

pd.set_option( 'display.max_columns' , 100 )

df = pd.read_csv('house/train.csv')

# -

# ## Data Description
#
#
# ### 부지 관련 정보
#     - MSSubClass : 집 타입 ( 복층이냐 최신식 이냐.. 등등 ) category ( str )
#     - MSZoning : 부지가 포함된 대상 지역의 용도 ( 농업용 용지, 상업지구 , 빌라냐 ... ) category ( str )
#
#     - LotFrontage : 부지와 도로가 맞닿아 있는 길이 ( 단위 feet ) 연속형 변수, 비율척도 ( float )
#     - LotArea : 부지 평방 피트 ( 단위 feet ) 연속형 변수, 비율척도 ( float )
#
#     - LotConfig : 부지 형태 ( FR2 frontage ..) category ( str )
#     - Lotshape : 부지 의 일반적인 모양? 형태 ( 규직적이냐 , 조금은 덜 규칙적이냐? ..) category ( str )
#     - LandContour : 부지 의 평탄도 ( 평평하냐 도로로 부터 조금 볼록하냐 ) category ( str )
#  
#
#     - Street : 부지에 인접한 도로 형태 ( 비포장, 포장 ) category ( str )
#     - Alley : 부지의 뒷골목의 형태 ( 포장, 비포장 ) category ( str )
#
#     - Utilitlies : 공공시설의 구비정도 ( 전기, 가스, 물.. ) category ( str )
#
#     - Landslope : 지면의 기울기정도 category ( str )
#     - Neighborhood : 가장 가까운 지역(타 지역에 인접 ) category ( str )
#     - Condition1 : 다양한 조건들 ( 어떤 교통편과 인접해 있는지 ) category ( str )
#     - Condition2 : Condition1외 추가 조건 / category ( str )
#
# -----------------------------------------------------
# ### 거주지 외관 정보
#
#     - BldgType : 거주지 타입 ( 독립 주거 형태 1층이냐 2층이냐? ) category *** ( str )
#     - dwelling : 주택 형식 ( duplex 복식구조 , townhouse 형 ) category ** ( str )
#     - HouseStyle : 주택의 스타일 ( 1층, 2층... ) category ( str ) 
#
# **지붕**
#
#     - RoofStyle : 지붕의 타입 ( 평평한 지붕 아치형 축사. ) category ( str )
#     - RoofMatl : 지붕 재질 ( 타일 , 간판 , 양피지?? ) category ( str )
#
#
# **베니어 ( 담장 )**
#
#     - MasVnrArea : 벽돌 베니어의 평방 피트 연속형변수 비율척도 ( float ) 
#     - MasVnrType : 벽돌 베니어 타입 ( 일반적인 벽돌 , 벽돌 모양들 ) category ( str ) 
#
# **집 외부 재질**
#
#     - Exterior1st : 집 외부 재질 ( 석면 외관.. ) category ( str )
#     - Exterior2nd : 집 외부 추가 재질 ( 1과동일, 추가적인 재질이 들어가있다면 명시 ) category ( str )
#
#
# ### 거주지 내부 정보
#
# **거주지 지상층 정보**
#
#     
#     - 1stFlrSF : 1층 평방 피트 / 연속형 변수 , 비율척도 ( float )
#     - 2stFlrSF : 2층 평방 피트 / 연속형 변수 , 비율척도 ( float )
#     - LowQualFinSF : 비사용공간의 평방피트(모든층) / 연속형 변수 , 비율척도 ( float )
#     - GrLivArea : 1층~ 부터의 평방 피트 / 연속형 변수 , 비율척도 ( float )
#
#     - FullBath : ground fullbathroom 유무 / category ( bool )
#     - HalfBath : ground halfbathroom 유무 / category ( bool )
#
#     - Bedroom : 지상층의 bedrooms 유무 ( 지하실은 포함하지 않는다 ) / category ( bool )
#     - Kitchen : 지상층의 주방 유무 / category ( bool )
#     - KitchenQual : 주방의 퀄리티 ( Ex , Gd ) / category ( bool )
#
#     - TotRmsAbvGrd : 지상층의 모든방 갯수 ( bedroom 제외 ) / 연속형 변수 , 비율척도 ( float )
#     - Functional : 집의 기능 / category ( str )
#
#     - Fireplaces : 벽난로 갯수 / 연속형 변수 , 비율척도 ( float )
#     - FireplaceQu : 벽난로 퀄리티 / category , 서열척도 ( str )
#     
#     
# **지상층 집 외부 시설 ( 차고 , 베란다 , deck )**
#
#     - GarageType : 차고 위치 ( 지상, 지하 , 2개 이상 , 빌트인 ) /category ** ( str )
#     - GarageYrBlt : 차고 건축 연도 / category ( str )
#
#     - GarageFinish : 차고 마감정도 / category ( str )
#
#     - GarageCars : 차고 차량 수용면적 / 연속형 변수 , 비율척도 ( float )
#     - GarageArea : 차고 전체 면적 / 연속형 변수 , 비율척도 ( float )
#     - GarageQual : 차고 퀄리티 / category , 서열척도 ( str )
#     - GarageCond : 차고 상태 / category , 서열척도 ( str )
#     - paveDrive : 차가 지나가는길 포장상태 / category ( bool )
#
#     - WoodDeckSF : wooddeck 평방 피트 / 연속형 변수 , 비율척도 ( float )
#     - OpenPorchSF : 개방 현관 평방피트 / 연속형 변수 , 비율척도 ( float )
#     - EnclosedPorch : 패쇄 현관 평방피트 / 연속형 변수 , 비율척도 ( float )
#     - 3SsnPorch : 3면이 뚫려있는 현관 평방피트 / 연속형 변수 , 비율척도 ( float )
#     - ScreenPorch : 차광이된 현관 평방피트 / 연속형 변수 , 비율척도 ( float )
#     - poolArea : pool 평방피트 / 연속형 변수 , 비율척도 ( float )
#
#     - poolQC : pool 의 퀄리티( Ex , Gd ..) /서열척도
#     - Fence : 펜스 퀄리티 / category 
#     
# **거주지 내부의 환경시설, 편의시설 정보**
#
#     - Heating : 난방형태 (마루난방, 가스기반 ) category **
#     - HeatingQC : 난방의 질 ( EX , Gd ) category , 서열척도 ( str )
#     - CentralAir : 에어컨디셔너 유무  /  category  ( bool )
#
#     - Electrical : 전기시스템 / category ** ( str ) 
#     - MiscFeature : 엘리베이터 2개의 차고, 기타 편의시설등 추가적인 공간 여부 ( Elev , Gar2 , shed ) / category ( str ) 
#     - MiscVal : 추가편의시설들의 가치 (doller) / 연속형 변수 , 비율척도 ( float )
#     
#
# **거주지의 지하실 정보**
#
#     - BsmtQual : 지하실의 높이 ( 인치기준으로 높이에 따라 서열 분류 ) category , 서열척도 ( str ) 
#
#     - BsmtFinSF1 : 지하실의 평방피트 / 연속형 변수 , 비율척도 ( float )
#     - BsmtFinSF2 : 지하실의 평방피트 / 연속형 변수 , 비율척도 ( float )
#     - BsmtUnfSF : 지하실의 자투리공간 평방피트 / 연속형 변수 , 비율척도 ( float )
#     - TotalBsmtSF : 전체 지하실의 평방피트 / 연속형 변수 , 비율척도 ( float )
#
#
#     - BsmtFullBath : 지하실 fullbathroom 유무 / category ( bool )
#     - BsmtHalfBath : 지하실 halfbathroom 유무 / category ( bool )
#
#     - BsmtCond : 전반적인 지하실 상태 ( Ex , Gd ) category , 서열척도 ( str )
#     - BsmtFinType1 : 지하실 용도  ( recreation용도냐 living용도냐 ) category ** ( str )
#     - BSmtExposure : 지하실 노출정도 ( Gd , No 노출되어 있지 않음 , NA 지하실이 없음  ) category ** ( str )
#     - BsmtFinType2 : 추가적인 지하실의 용도 ( BsmtFinType1외 추가적인 지하실이 있는경우 ) category ** ( str ) 
#     
# -------------------------
# ### 주택 거래 정보
#
#     - Mosold : 판매월 / 연속형 변수 , 비율척도 ( float )
#     - Yrsold : 판매년도 / 연속형 변수 , 비율척도 ( float )
#
#     - SaleType : 판매 방식 ( 보증 , 현금 , 대츨 , 신축구매.. ) / category ** ( str ) 
#     - SaleCondition : 판매조건 ( 일반적인 , 할인행사 , 교환 , 가족양도 , 정원과 같이 구매 ) / category ** ( str )
#
#
# **전반적인 평가**
#
#     - OverallQual : 전반적인 집의 모양 자제 에대한 평가 / 연속형 변수 , 등간척도 ( int ) 
#     - OverallCond : 집 조건에 대한 평가 / 연속형 변수 , 등간척도 ( int )
#     - YearBuilt : 건축 연식 / 연속형 변수 ( date ) 
#     - YearRmodAdd : 재건축 날짜 / 연속형 변수 ( date ) 
#
#     - ExterQual : 외부 형태 평가 ( Ex , Gd ) 연속형 변수 , 등간척도 ( int ) 
#     - ExterCond : 외부 상태와 재질에 대한 평가 / 연속형 변수 , 등간척도 ( int )
#     - Foundation : 집 기반의 타입 ( BrkTil , CBlock .. ) category ( str ) 
#
#
#
#
#

# # EDA 및 EF
# 예측변수 : 주택 가격 ( SalePrice )
# 설명변수 : 주택가격을 제외한 모든 속성들
#
# ### NULL 값 확인 및 처리

plt.figure( figsize = ( 10 , 10 ) )
sns.heatmap(df.isnull() , cmap = 'Blues_r')

# ### null 값을 가진 속성들
#
# 1. 풀장의 널값 --> 수영장을 소유한 주택이 적음을 알 수 있음. --> 'None'으로 채운다.
# 2. 추가 편의시설에 대한 정보 --> 추가 편의시설을 소유한 주택의 수가 적음 --> 'None'으로 채운다.
# 3. 뒷골목, 담장에 대한 정보 --> 없는 주택은 진짜 없는 주택.. --> 'None'으로 채운다.
# 4. 차고지 관련 데이터의 결측 --> 차고지가 없는 주택은 81채 --> 'None'으로 채운다.
# 5. 지하실의 관련 데이터 결측 --> 지하실이 없는 주택 38채 한개만 있는 주택은 1채 --> 'None'으로 채운다
# 6. 부지관련 ( LotFrontage ) 의 널값 처리 필요성

# 담장 타입에 대한 정보도 없으니 0으로 채운다.
df.MasVnrArea = df.MasVnrArea.fillna(0)
# 차고가 없는 주택  - GarageYrBlt
df.GarageYrBlt = df.GarageYrBlt.fillna(0)

# #### LotFrontage
#
# 부지와 맞닿아있다면, 부지의 형태와 유의미한 관계가 있다고 판단 가능 
# 부지의 성격에 따른 LotFrontage의 평균값

sns.violinplot( data = df[['LotConfig' , 'LotFrontage']] , x = 'LotConfig'  , y = 'LotFrontage')
plt.title( '부지 종류별 lotfrontage 의 분포도 ')
plt.show()

# > 부지의 유형에 맞게 부지의 유형 별로, 평방미터와, frontage 의 분포가 다름을 보인다. 이는 부지와 도로가 인접해 있는 길이가, 부지의 크기와 관계가 없을을 보여준다.
# 이를 고려하여 부지의 유형에 따라 그 유형에 맞는 평균 값으로 보간하도록 한다.

# LotFrontage 보간
tmp = df.groupby('LotConfig')['LotFrontage'].mean()
for i in tmp.index:
    df.loc[ (df.LotFrontage.isnull()) & (df.LotConfig == i) , 'LotFrontage' ] = tmp[i]

# #### RoofMatl
# - 지붕 재질 ( 타일 , 간판 , 양피지?? ) category ( str )
# - 'Tar&Grv','WdShngl','WdShake' 세게의 값을 제외한 나머지 는 하나로 통일

df.loc[ ~df.RoofMatl.isin( ['Tar&Grv','WdShngl','WdShake'] ) , 'RoofMatl' ] = 'CompShg'


# #### MasVnrType
# BrkFace , BrkCmn --> Brk으로 통일

df.loc[ df.MasVnrType.isin( ['BrkFace' , 'BrkCmn'] ) , 'MasVnrType'] = 'Brk'

# #### ExterCond , ExterQual
#
# 의미 중복. 둘다 외부에 대한 상태와 퀄리티를 나타내는 컬럼
#
# 예측 변수와 고려하였을때, ExterCond의 분포의 차이가 미미함
#
# 반면 ExterQual의 카테고리에 따라 가격분포가 나뉘며, Qual가 좋은데 상태가 보통인 데이터가 다수 존재
#
# 예측변수의 설명능력이 ExterQual이 ExterCond보다 높아 ExterQual 선택!

# #### BsmtExposure
#
# 의미 축소 설명변수의 분포가 작다.
#
# 노출의 정도를 나타내는 필드( 변수 ) 지만, 너무 구체적이고 예측변수간의 분포 또한 차이가 없다.
#
# 노출이 되어있는가 ( 'Gd' , 'Av' )
# 노출이 되어있지 않은가 ( 'No' , 'Mn' )
# 지하실이 없는가 ( 'None' ) 으로 변수 재정의

df.loc[ ~df.BsmtExposure.isin( ['No' ,'Mn', 'None'] ) , 'BsmtExposure'] = 'Yes'
df.loc[ df.BsmtExposure.isin( ['No' ,'Mn'] ) , 'BsmtExposure'] = 'No'

# #### 그외 데이터

# +
# 필요없는 변수 제거
tgt_cat_col = ['LotShape','LandContour','LotConfig','Neighborhood',
 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',
 'ExterQual','Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',
 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'FireplaceQu', 'GarageType', 'GarageFinish',
 'PavedDrive', 'PoolQC', 'SaleType', 'SaleCondition', 'Condition1', 'GarageQual',
 'GarageCars' , 'TotRmsAbvGrd' , 'BedroomAbvGr' , 'Fireplaces' , 'OverallCond' ,'HalfBath']

    
tgt_inter_col = ['FullBath','OverallQual',
'MSSubClass' , 'YearRemodAdd']

#  'EnclosedPorch','ScreenPorch','OpenPorchSF','3SsnPorch',
tgt_ratio_col = [ 'LotFrontage', 'GrLivArea',
 'MiscVal',  'LowQualFinSF', 'GarageArea', '2ndFlrSF',
 'WoodDeckSF', 'MasVnrArea', 'TotalBsmtSF', 'PoolArea', 'LotArea' , 'YearRemodAdd' ,'SalePrice']


# +
df = df.fillna('None')

sc = RobustScaler()


ohe = OneHotEncoder()

# 카테고리 함수는 onehotencodeing 하고, 나머지 컬럼은 로그를 씌워서 합치기.
# 로그를 씌운 이유는 , 정규성이 띄지 않기 때문에 
data = np.hstack( [ohe.fit_transform(df[tgt_cat_col]).toarray() , 
                   np.log(df[tgt_inter_col+tgt_ratio_col].replace(0 , 1)).values] )
data = sc.fit_transform(data)
# -

data

# #### 데이터 x , y 나누기 - split 

# +
print('data : ', data.shape)
X , y= data[:,:-1] , data[:,-1]

train_X , test_X , train_y , test_y = train_test_split( X , y , test_size = 0.18 , random_state = 10 )
# -

# ## 모델링

# #### LGBMRegressor

# +
from lightgbm import LGBMRegressor
model = LGBMRegressor( n_estimators= 512 , num_leaves = 128, learning_rate = 0.07, random_state= 12 ,
                     reg_alpha= 0.6 , reg_lambda= 8.8 , scale_po_weight = 2)
model.fit(train_X , train_y , verbose=True )


lgbm_pred = np.exp(model.predict(test_X)*sc.scale_[-1] + sc.center_[-1])
real = np.exp(test_y*sc.scale_[-1] + sc.center_[-1])
np.sqrt( ((lgbm_pred - real)**2).mean() )
# -

res = pd.DataFrame( np.dstack( [lgbm_pred , real])[0] , columns = ['predict', 'real'] ).sort_values('real')
plt.figure( figsize = ( 10 , 7 ))
sns.lineplot( data = res  , x = range( len(res) ) , y = 'predict', color= '#fb2e01', label = 'predict' )
sns.lineplot( data = res  , x = range( len(res) ) , y = 'real' , alpha = 0.7 , label = 'real' )
plt.legend()

# #### 선형회귀

# +
from sklearn.decomposition import PCA


sc = RobustScaler()
ohe = OneHotEncoder()
pca = PCA( n_components= 65 )

print(ohe.fit_transform(df[tgt_cat_col]).toarray().shape)
pca_data = pca.fit_transform( ohe.fit_transform(df[tgt_cat_col]).toarray() )

data = np.hstack( [pca_data , np.log(df[tgt_inter_col+tgt_ratio_col].replace(0 , 1)).values] )
data = sc.fit_transform(data)


print(data.shape)
X , y= data[:,:-1] , data[:,-1]

train_X , test_X , train_y , test_y = train_test_split( X , y , test_size = 0.18 , random_state = 10 )

# +
from sklearn.linear_model import ElasticNet

lr_model = ElasticNet(alpha = 0.0007 , l1_ratio = 0.0001, max_iter = 1000)
lr_model.fit( train_X , train_y )
lr_pred = np.exp(lr_model.predict(test_X)*sc.scale_[-1] + sc.center_[-1])
real = np.exp(test_y*sc.scale_[-1] + sc.center_[-1])
np.sqrt( ((lr_pred - real)**2).mean() )
# -

res = pd.DataFrame( np.dstack( [lr_pred , real])[0] , columns = ['predict', 'real'] ).sort_values('real')
plt.figure( figsize = ( 10 , 7 ))
sns.lineplot( data = res  , x = range( len(res) ) , y = 'predict', color= '#fb2e01', label = 'predict' )
sns.lineplot( data = res  , x = range( len(res) ) , y = 'real' , alpha = 0.7 , label = 'real' )
plt.legend()


