# NN-NBA

## How To Use
Open up python, ipython or jupyter notebook from the root directory of this project, then run

``` python
> from nnnba import *
> nnnba = NNNBA()
```

There are 8 models available:
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

## How It Works

#### Getting the data
The data is gathered from three different places: [NBA.com Stats](http://stats.nba.com), [Basketball Reference](http://basketball-reference.com/), and [hoopshype](http://hoopshype.com/). 

NBA.com was scraped using [nba_py](https://github.com/seemethere/nba_py) to gather player statistics (including advanced stats, misc stats, etc.). Basketball Reference was scraped using [basketballcrawler](https://github.com/andrewgiessel/basketballcrawler) to gather player age, current salary, etc. Then hoopshype was scraped to gather players' future salary. As Basketball Reference doesn't care to be scraped often, the information is saved in players.json. It's read when `prepare_data` is run, and combined with the nba_py and hoopshype data, then the data is stored in `raw_data.json`. 

#### Cleaning
As it turns out, a lot of the columns of data needed to be removed. Players who has played less than 15 games are removed as their high stats skewed the models.

#### Modeling
Each model uses their stats as an input, and salary as output. The idea is to fit the model to each player stats, and predict their value. The output is scaled from the min to the max contract price for 2017-18 season. An average is also done (and considered to be a separate model), which averages Bayes Ridge, Lasso, and ElasticNet output equally.

#### Player Value/Worth
Some of the methods used comes with coefficients that explains how the model works, and why specific players are ahead of others.  
For example, Ridge seems to favor Personal Fouls Drawn and Free Throws Made, so players who draws fouls and makes a lot of free throws would be deemed to be worth more. Thus, DeRozan (for all his ability to draw fouls and get to the line) is considered to be the most valuable. 
Linear Regression seem to value FG3M and FGM, so players with volume would be considered more valuable.

By comparing their calculated worth to their future salary, it is possible to find undervalued players as well. As the most valuable player is dependent on the model, the undervalued players are dependent on the model as well.

#### Shortcomings
It is important to note that this only analyzes stats based on past year performance, which is very isolated. It doesn't take into account team strength (though many models take wins into account), and potential. For example, Curry and Durant would have better stats on separate teams, and although their stats are still very impressive, the models don't take into account their stats are lowered than what it could've been. Therefore, their salary value is lowered. 

## Contribution
If you want to contribute, please see [CONTRIBUTING](/CONTRIBUTING.md) guidelines.
