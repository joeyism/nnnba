import json
import pandas as pd
import sys
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import normalize
from sklearn.pipeline import Pipeline
from sklearn import linear_model
from sklearn import ensemble
from sklearn.preprocessing import PolynomialFeatures
import xgboost as xgb

pd.set_option('display.max_columns', None)

with open("crawled_data/raw_data.json", "r") as data_file:
    raw_data = json.load(data_file)

#print (raw_data[0]["header"])
columns = raw_data[0]["header"]
#columns = ("GP","W","L","W_PCT","MIN","FGM","FGA","FG_PCT","FG3M","FG3A","FG3_PCT","FTM","FTA","FT_PCT","OREB","DREB","REB","AST","TOV","STL","BLK","BLKA","PF","PFD","PTS","PLUS_MINUS","DD2","TD3","GP_RANK","W_RANK","L_RANK","W_PCT_RANK","MIN_RANK","FGM_RANK","FGA_RANK","FG_PCT_RANK","FG3M_RANK","FG3A_RANK","FG3_PCT_RANK","FTM_RANK","FTA_RANK","FT_PCT_RANK","OREB_RANK","DREB_RANK","REB_RANK","AST_RANK","TOV_RANK","STL_RANK","BLK_RANK","BLKA_RANK","PF_RANK","PFD_RANK","PTS_RANK","PLUS_MINUS_RANK","DD2_RANK","TD3_RANK")

X_df = pd.DataFrame(columns=columns)
Y_df = pd.DataFrame()
names = pd.DataFrame(columns=["NAME"])


print ("Processing data")
for i, player in enumerate(raw_data):
    if "2016-17" in player["salaries"] and "2016-17" in player["stats"]:
        Y_df = Y_df.append([player["salaries"]["2016-17"]])
        X_df.loc[len(X_df)] = player["stats"]["2016-17"]
        names.loc[len(names)] = player["name"]
    else:
        continue

X_df = X_df.drop("W", 1).drop("L",1).drop("W_PCT",1)

def baseline_model():
    model = Sequential()
    model.add(Dense(98, input_dim=98, kernel_initializer='normal', activation='relu'))
    model.add(Dense(98, kernel_initializer='normal', activation='relu'))
    model.add(Dense(196, kernel_initializer='normal', activation='relu'))
    model.add(Dense(98, kernel_initializer='normal', activation='relu'))
    model.add(Dense(59, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal')) 
    model.compile(loss='mean_squared_error', optimizer='adam') 
    return model

X = X_df.values
Y = Y_df[0].values

#X = normalize(X)
models = { 
    "linear regression": linear_model.LinearRegression(),
    "ridge": linear_model.Ridge(alpha = 0.5),
    "bayesian ridge": linear_model.BayesianRidge(),
    "keras regressor": KerasRegressor(build_fn=baseline_model, nb_epoch=100, batch_size=5, verbose=0),
    "gradient boosting": ensemble.GradientBoostingRegressor(n_estimators= 500, max_depth= 2, learning_rate=0.01, loss="ls"),
    "lassocv": linear_model.LassoCV(alphas = [1, 0.1, 0.001, 0.0005]),
    "xgb": xgb.XGBRegressor(n_estimators=160, max_depth=2, learning_rate=0.1),
    "polynomial regression": Pipeline([('poly', PolynomialFeatures(degree=2)), ('linear', linear_model.LinearRegression())])
}
model_results = {}

for model_type, regr in models.items():
    this_results = names.copy()
    regr.fit(X,Y)
    
    
    results = regr.predict(X)
    this_results['WORTH'] = results
    
    diffY = Y - results
    this_results['SALARY_DIFF'] = diffY
    this_results = this_results.sort_values(by="SALARY_DIFF", ascending=False)

    model_results[model_type] = this_results
    print ("Finished " + model_type)

def findUndervalued(model_type="linear regression"):
    names = model_results[model_type]
    print (names.loc[names["SALARY_DIFF"] < 0])

def findPlayerWorth(player_name, model_type="linear regression"):
    names = model_results[model_type]
    idx = names[names["NAME"] == player_name].index[0]
    print ("\nPaid: " + '${:,.2f}'.format(float(Y[idx])) + "\tWorth: " + '${:,.2f}'.format(float(names["WORTH"][idx])) + "\n")
    findPlayerStats(player_name)

def findPlayerStats(player_name):
    print (X_df.loc[names["NAME"] == player_name, ])

def findBestPlayer(model_type="linear regression"):
    names = model_results[model_type]
    print (names.sort_values(by="WORTH")
)


def showAvailableModels():
    for model in models:
        print (model)
