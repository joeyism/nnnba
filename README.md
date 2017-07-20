# NN-NBA

## How To Use
Open up python, ipython or jupyter notebook from the root directory of this project, then run

``` python
> from nnnba import *
> nnnba = NNNBA()
```

There are 4 models available:
* [linear regression](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
* [ridge](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)
* [bayesian ridge](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.BayesianRidge.html)
* [keras regressor](https://keras.io/scikit-learn-api/)

To find undervalued players, run

``` python
> nnnba.findUndervalued()
```

where `model_type` is one of the models described above

To find a player's worth in each model, run

``` python
> nnnba.findPlayerWorth(player_name)
```

where `model_type` is one of the models described above, and `player_name` is the player's name

### For example

``` python
> nnnba.findPlayerWorth("Giannis Antetokounmpo")
```


