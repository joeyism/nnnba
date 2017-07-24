import json
import pandas as pd
import sys
import numpy as np
from logger import *

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.preprocessing import normalize
from sklearn.pipeline import Pipeline
from sklearn import linear_model
from sklearn import ensemble
from sklearn.preprocessing import PolynomialFeatures
from sklearn import svm
from scipy.stats import skew
import xgboost as xgb
import prepare_data

pd.set_option('display.max_columns', None)


class NNNBA:

    ridge_init_alpha = [0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6, 10, 30, 60]
    lasso_init_alpha = [0.0001, 0.0003, 0.0006, 0.001, 0.003, 0.006, 0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1]
    elasticnet_init = { 
        "l1_ratio":[0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1],
        "alpha":[0.0001, 0.0003, 0.0006, 0.001, 0.003, 0.006, 0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6]
    }

    def __realpha__(self, alpha):
        return [alpha * .6, alpha * .65, alpha * .7, alpha * .75, alpha * .8, alpha * .85, alpha * .9, alpha * .95, alpha, alpha * 1.05, alpha * 1.1, alpha * 1.15, alpha * 1.25, alpha * 1.3, alpha * 1.35, alpha * 1.4]

    def __reratio__(self, ratio):
        return [ratio * .85, ratio * .9, ratio * .95, ratio, ratio * 1.05, ratio * 1.1, ratio * 1.15]

    def __baseline_model__():
        model = Sequential()
        model.add(Dense(83, input_dim=83, kernel_initializer='normal', activation='relu'))
        model.add(Dense(41, kernel_initializer='normal', activation='relu'))
        model.add(Dense(83, kernel_initializer='normal', activation='relu'))
        model.add(Dense(41, kernel_initializer='normal', activation='relu'))
        model.add(Dense(20, kernel_initializer='normal', activation='relu'))
        model.add(Dense(1, kernel_initializer='normal')) 
        model.compile(loss='mean_squared_error', optimizer='adam') 
        return model



    models = { 
        "linear regression": linear_model.LinearRegression(fit_intercept=True),
        "ridge": linear_model.RidgeCV(alphas = ridge_init_alpha, fit_intercept=True),
        "lasso": linear_model.LassoCV( alphas = lasso_init_alpha, max_iter = 5000, cv = 10, fit_intercept = True),
        "keras regressor": KerasRegressor(build_fn=__baseline_model__, nb_epoch=100, batch_size=5, verbose=0),
        "xgb": xgb.XGBRegressor(n_estimators=1500, max_depth=2, learning_rate=0.01),
        #"elasticnet": linear_model.ElasticNetCV(l1_ratio = elasticnet_init["l1_ratio"], alphas = elasticnet_init["alpha"], max_iter = 1000, cv = 3),
        #"svr": svm.SVR(kernel="linear", C=1e3)
    }

    default_model_type = "linear regression"

    def __remodel__(self, model_type, regr, X):
        if model_type == "ridge":
            alpha = regr.alpha_
            regr = linear_model.RidgeCV(alphas = self.__realpha__(alpha), cv = 10)
        elif model_type == "lasso":
            alpha = regr.alpha_
            regr = linear_model.LassoCV(alphas = self.__realpha__(alpha), max_iter = 5000, cv = 10)
        elif model_type == "elasticnet":
            alpha = regr.alpha_
            ratio = regr.l1_ratio_
            regr = linear_model.ElasticNetCV(l1_ratio = self.__reratio__(ratio), alphas = self.elasticnet_init["alpha"], max_iter = 1000, cv = 3)

        regr.fit(X, self.Y)
        return regr
        

    def __init__(self):
        with open("crawled_data/raw_data.json", "r") as data_file:
            raw_data = json.load(data_file)

        columns = raw_data[0]["header"]
        positions = ["Point Guard", "Center", "Power Forward", "Shooting Guard", "Small Forward"]
        for i, val in enumerate(positions):
            positions[i] = (val, i)
        positions_convert = dict(positions)

        self.X_df = pd.DataFrame(columns=columns)
        Y_df = pd.DataFrame()
        age = []
        positions = []
        names = pd.DataFrame(columns=[ "NAME", "PROJECTED_SALARIES" ])


        logger.debug("Processing data")
        for i, player in enumerate(raw_data):
            if "2016_17" in player["salaries"] and "2016-17" in player["stats"]:
                Y_df = Y_df.append([player["salaries"]["2016_17"]])
                self.X_df.loc[len(self.X_df)] = player["stats"]["2016-17"]
                age.append(player["age"])

                # put in their own binary columns
                position_num = []
                for position in player["positions"]:
                    position_num.append(positions_convert[position])
                positions.append(sum(position_num)/len(position_num))

                projected_salaries = 0
                try:
                    projected_salaries = player["projected_salaries"][0]
                except:
                    pass
                names.loc[len(names)] = [ player["name"], projected_salaries ]
            else:
                continue

        self.X_df = pd.concat([self.X_df, pd.Series(age, name="AGE"), pd.Series(positions, name="POSITION")], axis=1) 

        # normalize skew
        # skewed = self.X_df.apply(lambda x: skew(x)) 
        # skewed_columns = skewed[skewed > 0.75].index # inspired by apapiu
        # self.X_df[skewed_columns] = np.log1p(self.X_df[skewed_columns])


        X = self.X_df.values

        #X = normalize(X) #Normalize X decreases worth

        self.Y = np.log1p(Y_df[0].values) # y = log(1+y)
        self.model_results = {}
        self.names = names

        for model_type, regr in self.models.items():
            logger.debug("Started  " + model_type)
            this_results = names.copy()
            regr.fit(X, self.Y)

            regr = self.__remodel__(model_type, regr, X)
            
            results = np.expm1(regr.predict(X)) # y = exp(y) - 1
            this_results['WORTH'] = results
            
            diffY = this_results["PROJECTED_SALARIES"].values - results
            this_results['SALARY_DIFF'] = diffY
            this_results = this_results.sort_values(by="SALARY_DIFF", ascending=False)
            
            self.models[model_type] = regr
            self.model_results[model_type] = this_results
            logger.debug("Finished " + model_type)


    def findUndervalued(self, model_type=default_model_type):
        names = self.model_results[model_type]
        print(names.loc[(names["SALARY_DIFF"] < 0) & (names["PROJECTED_SALARIES"] > 0)])
    
    def findPlayerWorth(self, player_name, model_type=default_model_type):
        names = self.model_results[model_type]
        idx = names[names["NAME"] == player_name].index[0]
        print("\nPaid: " + '${:,.2f}'.format(float(self.Y[idx])) + "\tFuture Salary: " + '${:,.2f}'.format(float(self.names["PROJECTED_SALARIES"][idx])) + "\tWorth: " + '${:,.2f}'.format(float(names["WORTH"][idx])) + "\n")
        self.findPlayerStats(player_name, trim=True)
    
    def findPlayerStats(self, player_name, trim=False):
        columns = self.X_df.columns
        if trim:
            columns = columns[:30]
        print(self.X_df.loc[self.names["NAME"] == player_name, columns])
    
    def findMostValuablePlayer(self, model_type=default_model_type):
        names = self.model_results[model_type]
        print(names.sort_values(by="WORTH")
    )
    
    def showAvailableModels(self):
        for model in self.models:
            print(model)

    def getCoefFromModel(self, model_name):
        return pd.DataFrame(self.models[model_name].coef_, index=self.X_df.columns, columns=["coef"]).sort_values(by="coef")

def get_data(parallel=True):
    prepare_data.start(parallel=parallel)
