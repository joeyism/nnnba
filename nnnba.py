import json
import os
import pandas as pd
import sys
import numpy as np
from .logger import *
import logging

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
from . import prepare_data

pd.set_option('display.max_columns', None)


class NNNBA:
    """
    NNNBA class, which contains all the calculated information
    """

    default_model_type = "lasso"
    assumed_max_salary = 35350000.0


    all_player_names = []

    __threshold_per_col = {"OFF_RATING": 12, "PIE":0.11, "NET_RATING": 18, "GP": 50, "DEF_RATING": 7, "USG_PCT": 0.12, "FGA": None, "FGM": None, "FG3A": None, "PTS": None, "FTM": None, "FGM": None, "REB_PCT": None, "AGE": 4}

    __outlier_cols_upper = [] #["OFF_RATING", "PIE", "NET_RATING", "USG_PCT", "PTS"]
    __outlier_cols_lower = [] #["DEF_RATING"]

    __ridge_init_alpha = [0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6, 10, 30, 60]
    __lasso_init_alpha = [0.0001, 0.0003, 0.0006, 0.001, 0.003, 0.006, 0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1]
    __elasticnet_init = { 
        "l1_ratio":[0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1],
        "alpha":[0.0001, 0.0003, 0.0006, 0.001, 0.003, 0.006, 0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6]
    }

    def __realpha__(self, alpha):
        """
        Function to recalculate alpha
        """
        return [alpha * .6, alpha * .65, alpha * .7, alpha * .75, alpha * .8, alpha * .85, alpha * .9, alpha * .95, alpha, alpha * 1.05, alpha * 1.1, alpha * 1.15, alpha * 1.25, alpha * 1.3, alpha * 1.35, alpha * 1.4]

    def __reratio__(self, ratio):
        """
        Function to recalculate ratio
        """
        return [ratio * .85, ratio * .9, ratio * .95, ratio, ratio * 1.05, ratio * 1.1, ratio * 1.15]

    def __baseline_model__():
        """
        Base Neural Network model
        """
        input=39
        model = Sequential()
        model.add(Dense(input, input_dim=input, kernel_initializer='normal', activation='relu'))
        model.add(Dense(int(input/2), kernel_initializer='normal', activation='relu'))
        model.add(Dense(input, kernel_initializer='normal', activation='relu'))
        model.add(Dense(int(input/2), kernel_initializer='normal', activation='relu'))
        model.add(Dense(int(input/4), kernel_initializer='normal', activation='relu'))
        model.add(Dense(1, kernel_initializer='normal')) 
        model.compile(loss='mean_squared_error', optimizer='adam') 
        return model

    def __idx_of_median_outlier__(self, col, threshold=None, upper_outlier=True): #may need threshold=2
        """
        Find index of outlier based on distance from median
        Distance from median = threshold, which is either passed in or calculated as a function of std from the passed in data
        """
        if threshold is None:
            threshold = col.std()*2.5
        logger.debug("median: " + str(col.median()) + " threshold: " + str(threshold))
        diff = col - col.median()
        if upper_outlier:
            outlier = diff > threshold
        else:
            outlier = -1*diff > threshold
        return list(outlier.index[outlier])


    models = { 
        "linear regression": linear_model.LinearRegression(fit_intercept=True),
        "ridge": linear_model.RidgeCV(alphas = __ridge_init_alpha, fit_intercept=True),
        "lasso": linear_model.LassoCV( alphas = __lasso_init_alpha, max_iter = 5000, cv = 10, fit_intercept = True),
        "bayes ridge": linear_model.BayesianRidge(),
        "keras regressor": KerasRegressor(build_fn=__baseline_model__, nb_epoch=100, batch_size=5, verbose=0),
        "xgb": xgb.XGBRegressor(n_estimators=1500, max_depth=2, learning_rate=0.01),
        "elasticnet": linear_model.ElasticNetCV(l1_ratio = __elasticnet_init["l1_ratio"], alphas = __elasticnet_init["alpha"], max_iter = 1000, cv = 3),
        "theilsen": linear_model.TheilSenRegressor(),
        "polynomial": Pipeline([('poly', PolynomialFeatures(degree=2)), ('linear', linear_model.LinearRegression(fit_intercept=True))])
    }


    def __remodel__(self, model_type, regr, __X_train, __Y_train):
        """
        Function to retrain certain models based on optimal alphas and/or ratios
        """
        if model_type == "ridge":
            alpha = regr.alpha_
            regr = linear_model.RidgeCV(alphas = self.__realpha__(alpha), cv = 10)
        elif model_type == "lasso":
            alpha = regr.alpha_
            regr = linear_model.LassoCV(alphas = self.__realpha__(alpha), max_iter = 5000, cv = 10)
        elif model_type == "elasticnet":
            alpha = regr.alpha_
            ratio = regr.l1_ratio_
            regr = linear_model.ElasticNetCV(l1_ratio = self.__reratio__(ratio), alphas = self.__elasticnet_init["alpha"], max_iter = 1000, cv = 3)

        regr.fit(__X_train, __Y_train)
        return regr

    def __normalize_salary__(self, col, max_salary=assumed_max_salary): # scales out to max contract; max taken from https://www.hoopsrumors.com/2017/05/nba-maximum-salary-projections-for-201718.html
        """
        Function to normalize salary so that the max is maximum salary possible, as yoy max salary changes
        """
        min_salary = min(col)
        local_max_salary = max(col)
        return max_salary - (local_max_salary - col)/(local_max_salary - min_salary) * (max_salary - min_salary)
        

    def __init__(self, debug=False):
        logger.setLevel( logging.DEBUG if debug else logging.ERROR)
        fn = os.path.join(os.path.dirname(__file__), "crawled_data/raw_data.json")
        with open(fn, "r") as data_file:
            raw_data = json.load(data_file)

        columns = raw_data[0]["header"]
        unique_columns = list(set( raw_data[0]["header"]))
        position_names = ["Point Guard", "Shooting Guard", "Small Forward", "Power Forward", "Center"]
        positions = []

        for i, val in enumerate(position_names):
            positions.append((val, i))
        positions_convert = dict(positions)

        self.X_df = pd.DataFrame(columns=columns)
        Y_df = pd.DataFrame(columns=["SALARIES"])
        age = []
        positions_df = pd.DataFrame(columns = position_names)
        names = pd.DataFrame(columns=[ "NAME", "PROJECTED_SALARIES" ])


        logger.debug("Processing data")
        for i, player in enumerate(raw_data):
            if "2016_17" in player["salaries"] and "2016-17" in player["stats"]:
                Y_df.loc[len(Y_df)] = player["salaries"]["2016_17"]
                self.X_df.loc[len(self.X_df)] = player["stats"]["2016-17"]
                age.append(player["age"])

                positions_df.loc[len(positions_df)] = [0,0,0,0,0]
                for position in player["positions"]: #TODO: fix positions
                    positions_df[position][len(positions_df)] = 1

                projected_salaries = 0
                try:
                    projected_salaries = player["projected_salaries"][0]
                except:
                    pass
                names.loc[len(names)] = [ player["name"], projected_salaries ]
            else:
                continue

        for col in []:
            try:
                self.X_df[col] = np.tanh(self.X_df[col])
            except:
                pass

        self.X_df = self.X_df.T.drop_duplicates().T
        self.X_df = pd.concat([self.X_df, pd.Series(age, name="AGE"), positions_df], axis=1) 

        self.X_df = self.X_df.drop(["FGA", "L", "AGE", "PCT_TOV", "BLKA", "AST_PCT", "AST_RATIO", "OREB_PCT", "DREB_PCT", "REB_PCT", "TM_TOV_PCT", "PACE", "OPP_PTS_OFF_TOV", "OPP_PTS_FB", "OPP_PTS_PAINT", 'OPP_PTS_2ND_CHANCE', 'OPP_PTS_FB', 'PCT_FGA_2PT', 'PCT_FGA_3PT', 'PCT_PTS_2PT', 'PCT_PTS_2PT_MR', 'PCT_PTS_3PT', 'PCT_PTS_FB', 'PCT_PTS_FT', 'PCT_PTS_OFF_TOV','PCT_PTS_PAINT', 'PCT_AST_2PM', 'PCT_UAST_2PM', 'PCT_AST_3PM', 'PCT_UAST_3PM', 'PCT_AST_FGM', 'PCT_UAST_FGM', 'PCT_FGM', 'PCT_FGA','PCT_FG3M', 'PCT_FG3A', 'PCT_FTM', 'PCT_FTA', 'PCT_OREB', 'PCT_DREB','PCT_REB', 'PCT_AST', 'PCT_STL', 'PCT_BLK', 'PCT_BLKA', 'PTS_OFF_TOV', 'PTS_FB', 'PTS_PAINT'], 1)

        logger.debug("Columns: " + ", ".join(self.X_df.columns))
        # remove players who's played less than 15 games
        idx_of_lt_gp = self.X_df.index[(self.X_df["GP"] < 15)]
        self.X_df = self.X_df.drop(idx_of_lt_gp)
        Y_df = Y_df.drop(idx_of_lt_gp)
        age = pd.Series(age).drop(idx_of_lt_gp)
        positions_df = positions_df.drop(idx_of_lt_gp)
        names = names.drop(idx_of_lt_gp)


        
        # Remove outliers
        logger.debug("Remove outliers")

        X_train = self.X_df.copy()
        Y_train = Y_df.copy()
        logger.debug("No of rows before removing outliers: " + str(X_train.shape[0]))
        to_be_dropped = []
        ## remove upper
        for col in self.__outlier_cols_upper:
            logger.debug(col)
            idx_of_median_outlier = self.__idx_of_median_outlier__(X_train[col], self.__threshold_per_col[col])
            logger.debug(col +" should drop "+ ", ".join(names["NAME"][idx_of_median_outlier].values))
            to_be_dropped = to_be_dropped + idx_of_median_outlier

        ## remove lower
        for col in self.__outlier_cols_lower:
            logger.debug(col)
            idx_of_median_outlier = self.__idx_of_median_outlier__(X_train[col], self.__threshold_per_col[col], upper_outlier=False)
            logger.debug(col +" should drop "+ ", ".join(names["NAME"][idx_of_median_outlier].values))
            to_be_dropped = to_be_dropped + idx_of_median_outlier

                   
        to_be_dropped = list(set(to_be_dropped))
        logger.debug("Outliers: " + ", ".join(names["NAME"][to_be_dropped].values))
        X_train = X_train.drop(to_be_dropped)
        Y_train = Y_train.drop(to_be_dropped)
        logger.debug("No of rows after removing outliers: " + str(X_train.shape))
        logger.debug("No of rows after removing outliers: " + str(Y_train.shape))



        __X_train = X_train.values # training data only includes non-rookies
        __Y_train = np.log1p(Y_train["SALARIES"].values) # y = log(1+y)

        self.Y_df = Y_df
        self.model_results = {}
        self.names = names

        for model_type, regr in self.models.items():
            logger.debug("Started  " + model_type)
            this_results = names.copy()
            regr.fit(__X_train, __Y_train)

            regr = self.__remodel__(model_type, regr, __X_train, __Y_train)
            
            results = self.__normalize_salary__(np.expm1(regr.predict(self.X_df.values))) # y = exp(y) - 1
            this_results['WORTH'] = results
            
            diffY = this_results["PROJECTED_SALARIES"].values - results
            this_results['SALARY_DIFF'] = diffY
            this_results = this_results.sort_values(by="SALARY_DIFF", ascending=False)
            
            self.models[model_type] = regr
            self.model_results[model_type] = this_results
            logger.debug("Finished " + model_type)

        #get avg
        this_results = self.model_results["linear regression"].copy()
        this_results["WORTH"] = self.__normalize_salary__((1.*self.model_results["bayes ridge"]["WORTH"] + 1.*self.model_results["lasso"]["WORTH"] + 1.*self.model_results["elasticnet"]["WORTH"])/3)
        diffY = this_results["PROJECTED_SALARIES"].values - this_results["WORTH"]
        this_results['SALARY_DIFF'] = diffY
        self.model_results["avg"] = this_results

        # add all_player_names
        self.all_player_names = list(names["NAME"].values)


    def getUndervalued(self, model_type=default_model_type):
        names = self.model_results[model_type]
        return names.loc[(names["SALARY_DIFF"] < 0) & (names["PROJECTED_SALARIES"] > 0)]
    
    def getPlayerValue(self, player_name, model_type=default_model_type):
        names = self.model_results[model_type]
        idx = names[names["NAME"] == player_name].index[0]
        paid = float(self.Y_df.loc[idx]["SALARIES"])
        projected_salaries = float(self.names["PROJECTED_SALARIES"][idx])
        worth = float(names["WORTH"][idx])

        self.getPlayerStats(player_name, trim=True)
        return {"paid": paid, "projected_salaries": projected_salaries, "worth": worth}
    
    def getPlayerStats(self, player_name, trim=False):
        columns = self.X_df.columns
        if trim:
            columns = columns[:30]
        return self.X_df.loc[self.names["NAME"] == player_name, columns]
    
    def getMostValuablePlayers(self, model_type=default_model_type):
        names = self.model_results[model_type]
        return names.sort_values(by="WORTH")
    
    
    def showAvailableModels(self):
        available_model = []
        for model in self.model_results:
            available_model.append(model)
        return available_model

    def getPlayerNameByIndex(self, index):
        return self.names[self.name.index == index]

    def getCoefFromModel(self, model_type= default_model_type):
        return pd.DataFrame(self.models[model_type].coef_, index=self.X_df.columns, columns=["coef"]).sort_values(by="coef")

    def plotXCol(self, col_name, X = None):
        import matplotlib.pyplot as plt
        if X is None:
            X = self.X_df.sort_values(by=col_name)[col_name].values
        plt.figure()
        plt.scatter(range(len(X)), X)
        plt.show()


def get_data(parallel=True):
    prepare_data.start(parallel=parallel)
