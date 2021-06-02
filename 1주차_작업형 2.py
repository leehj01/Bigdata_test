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
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier

import warnings
warnings.filterwarnings('ignore')
# -

# ### 데이터 불러오기

wine_train = pd.read_csv('wine/train.csv')
wine_test = pd.read_csv('wine/test.csv')

wine_train.head()

# ### 1. feature engineering

#  white, red 를 0, 1 로 변환
wine_train['type'] =  wine_train['type'].map({'white':0, 'red':1}).astype(int)
wine_test['type'] = wine_test['type'].map({'white':0,'red':1}).astype(int)

# ### 2. train_x, train_y 데이터 나누기 

# train_x , trian_y 나누기 
x = wine_train.drop(['index', 'quality'], axis = 1)
y = wine_train['quality']
test_x = wine_test.drop('index', axis = 1)
print(train_x.shape, train_y.shape, test_x.shape)

# validation set , train set 구분
x_train, x_val , y_train, y_val = train_test_split( x, 
                                                    y,
                                                    test_size = 0.2,
                                                    shuffle = True,
                                                    random_state = 10)

print(x_train.shape, x_val.shape, y_train.shape, y_val.shape)

# ### 3. modeling 및 평가

# #### 1. XGBClassifier

# +
model = xgb.XGBClassifier(
                            max_depth = 10,
                            learning_rate=0.01,
                            n_estimators=100,
                           random_state = 0
                    )
model.fit(x_train, y_train)

y_pred = model.predict(x_val)

print(classification_report(y_val, y_pred))
# -

# #### RandomForestClassifier

# RandomForestClassifier(
#         n_estimators : 결정트리 개수
#         max_features : 결정 트리의 max_features 파라미터 
#         )

# +
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(x_train, y_train)

y_pred = rf_model.predict(x_val)
y_pred

pd.crosstab(y_val, y_pred)


print(classification_report(y_val, y_pred))
# -

# #### SVC

# +
svm_clf = SVC(random_state=42)
svm_clf.fit(x_train, y_train)

y_pred = svm_clf.predict(x_val)
pd.crosstab(y_val, y_pred)

print(classification_report(y_val, y_pred))
# -

# ## ENSEMBLE

# +
from sklearn.ensemble import VotingClassifier

voting_model = VotingClassifier(estimators=[('rf', rf_model), ('xgb', model), ('svm', svm_clf)],
                               voting='hard')

# votingclassifier 은 train 을 그냥 넣어주면 오류남으로 numpy 형으로 변환
voting_model.fit(x_train.to_numpy(), y_train)

voting_model.get_params()

y_pred = voting_model.predict(x_val.to_numpy())
pd.crosstab(y_val, y_pred)

print(classification_report(y_val, y_pred))
# -




