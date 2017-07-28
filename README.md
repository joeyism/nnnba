# NN-NBA

## How To Use
Open up python, ipython or jupyter notebook from the root directory of this project, then run

``` python
> from nnnba import *
> nnnba = NNNBA()
```

There are 4 models available:
* [linear regression](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
* [lasso](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoCV.html)
* [ridge](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeCV.html)
* [bayesian ridge](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.BayesianRidge.html)
* [keras regressor](https://keras.io/scikit-learn-api/)
* [XGBoost](http://xgboost.readthedocs.io/en/latest/python/python_api.html)
* [ElasticNet](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html)
* [TheilSen](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.TheilSenRegressor.html#sklearn.linear_model.TheilSenRegressor)


The default model (which can be set by `nnnba.default_model_type`) is **lasso**.

### Most Valuable Players
To find undervalued players, run

``` python
> nnnba.getMostValuablePlayers()
```

where `model_type` is one of the models described above.


### Undervalued Players
To find undervalued players, run

``` python
> nnnba.getUndervalued()
```

where `model_type` is one of the models described above.


### Player Value by Model
To find a player's value in each model, run

``` python
> nnnba.getPlayerValue(player_name)
```

where `model_type` is one of the models described above, and `player_name` is the player's name

### For example

``` python
> nnnba.getPlayerValue("Giannis Antetokounmpo")
```


