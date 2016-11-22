Monotonicity Constraint in Xgboost
===============================

Fitting a model with a high accuracy is great, but is usually not enough. Quite often, we also want a model to be simple and interpretable. An example of such an interpretable model is a linear regression, for which the fitted coefficient of a variable means holding other variables as fixed, how the response variable changes with respect to the predictor. For a linear regression, this relationship is also **monotonic**: the fitted coefficient is either positive or negative.

##Model Monotonicity: An Example

Model monotonicity is useful in the real-world too. For example, when you apply for a credit card but were turned down, the bank usually tells you reasons (that you mostly don't agree with) why the decision is made. You may hear things like your previous credit card balances are too high, etc. In fact, this means that the bank's approval algorithm has a monotonically increasing relationship between an applicant's credit card balance and his / her risk. Your risk score is penalized because of a higher-than-average card balance.

If the underlying model is not monotonic, you may well find someone with a credit card balance $100 higher than you but otherwise identical credit profiles getting approved. To some extent, forcing the model monotonicity reduces overfitting. For the case above, it may improve fairness.

##Beyond Linear Models

It is possible, at least approximately, to force the model monotonicity constraint in a non-linear model as well. For a tree-based model, if for each split of a particular variable we require the right daughter node's average value to be higher than the left daughter node (otherwise the split will not be made), then approximately the variable's relationship with the dependent variable is monotonically increasing; and vise versa.

This monotonicity constraint feature has been implemented **`Xgboost`**. Below is a simple tutorial in Python.

##Tutorial for Xgboost

The California Housing dataset [1] is used for this tutorial. This dataset consists of 20,460 observations. Each observation represents a neighborhood in California. The response variable is the median house value of a neighborhood. Predictors include median income, average house occupancy, and location etc. of that neighborhood.

First we load the data. If you use **`scikit-learn`**, you just need to do:

```python

from sklearn.datasets.california_housing import fetch_california_housing
from sklearn.model_selection import train_test_split

cal_housing = fetch_california_housing()
print cal_housing.feature_names #['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
print cal_housing.data.shape #(20640, 8)
```

To start, we use a single feature "the median income" to predict the house value. We first split the data into training and testing datasets. Then we use a 5-fold cross-validation and early-stopping on the training dataset to determine the best number of trees. Last we use the entire training set to train the model and evaluate its performance on the testset.

```python
import numpy as np
import xgboost as xgb

X_train, X_test, y_train, y_test = train_test_split(cal_housing.data,
                                                    cal_housing.target,
                                                    test_size=0.4,
                                                    random_state=123)
feature_names = ['MedInc']
feature_ids = [cal_housing.feature_names.index(f) for f in feature_names]

dtrain = xgb.DMatrix(X_train[:, feature_ids].reshape(len(X_train), len(feature_ids)), label = y_train)
dtest =  xgb.DMatrix(X_test[:, feature_ids].reshape(len(X_test), len(feature_ids)), label = y_test)

# Setting variable monotonicity constraints
# 0: no constraint, 1: positive, -1: negative
feature_monotones = [0] * (len(feature_names))

params = {'max_depth': 2,
          'eta': 0.1,
          'silent': 1,
          'nthread': 2,
          'seed': 0,
          'eval_metric': 'rmse',

          # E.g. fitting three features with positive, negative and no constraint
          # 'monotone_constraints': (1,-1,0)
          'monotone_constraints': '(' + ','.join([str(m) for m in feature_monotones]) + ')'
         }

# Use CV to find the best number of trees
bst_cv = xgb.cv(params, dtrain, 500, nfold = 5, early_stopping_rounds=10)

# Train on the entire training set, evaluate performances on the testset
evallist  = [(dtrain, 'train'), (dtest, 'eval')]
evals_result = {}
bst = xgb.train(params, dtrain, num_boost_round = bst_cv.shape[0], evals_result = evals_result, evals = evallist,  verbose_eval = False)

```

Notice the model parameter `'monotone_constraints'`. This is where the monotonicity constraints are set in **`Xgboost`**. For now we set `'monotone_constraints': (0)`, which means a single feature without constraint.

Let's take a look at the model performances:

```python
print 'Number of boosting rounds %d,\
       Training RMSE: %.4f, \
       Testing RMSE: %.4f' % \
       (
        len(evals_result['train']['rmse']),
        evals_result['train']['rmse'][-1],
        evals_result['eval']['rmse'][-1]
        )
```
```python
Number of boosting rounds 56,       Training RMSE: 0.8167,        Testing RMSE: 0.8287
```

We can also check the relationship between the feature (median income) and the dependent variable (median house value):

```python
def partial_dependency(bst, X, y, feature_ids = [], f_id = -1):

    """
    Calculate the dependency (or partial dependency) of a response variable on a predictor (or multiple predictors)
    1. Sample a grid of values of a predictor.
    2. For each value, replace every row of that predictor with this value, calculate the average prediction.
    """

    X_temp = X.copy()

    grid = np.linspace(np.percentile(X_temp[:, f_id], 0.1),
                       np.percentile(X_temp[:, f_id], 99.5),
                       50)
    y_pred = np.zeros(len(grid))

    if len(feature_ids) == 0 or f_id == -1:
        print 'Input error!'
        return
    else:
        for i, val in enumerate(grid):

            X_temp[:, f_id] = val
            data = xgb.DMatrix( X_temp[:, feature_ids].reshape( (len(X_temp), len(feature_ids)) ) )

            y_pred[i] = np.average(bst.predict(data, ntree_limit = bst.best_ntree_limit))

    return grid, y_pred
```

```python
import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline

for f in feature_names:

    grid, y_pred = partial_dependency(bst,
                                      X_train,
                                      y_train,
                                      feature_ids = feature_ids,
                                      f_id = cal_housing.feature_names.index(f)
                                     )

    fig, ax = plt.subplots()
    fig.set_size_inches(5, 5)
    plt.subplots_adjust(left = 0.17, right = 0.94, bottom = 0.15, top = 0.9)

    ax.plot(grid, y_pred, '-', color = 'red', linewidth = 2.5, label='fit')
    ax.plot(X_train[:, cal_housing.feature_names.index(f)], y_train, 'o', color = 'grey', alpha = 0.01)

    ax.set_xlim(min(grid), max(grid))
    ax.set_xlabel(f, fontsize = 10)
    ax.set_ylabel('Partial Dependence', fontsize = 12)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc = 'best', fontsize = 12)
```

Here we create a helper function `partial_dependency` to calculate the variable dependency or partial dependency for an arbitrary model. The partial dependency [2] describes that when other variables fixed, how the average response depends on a predictor.

Without any monotonicity constraint, the relationship between the median income and median house value looks like this:

![One feature, no constraint](https://github.com/XiaoxiaoWang87/xgboost_mono_test/blob/master/no_constraint_one_feature.png)

One can see that at very low income and income around 10 (times its unit), the relationship between median income and house value is not strictly monotonic.

You may be able to find some explanations for this non-monotonic behavior (e.g. feature interactions). In some cases, it may even be a real effect which still holds true after more features are fitted. If you are very convinced about that, I suggest you not enforce any monotonic constraint on the variable, otherwise important insights may be ignored. But when the non-monotonic behavior is purely because of noise, setting monotonic constraints can reduce overfitting.

For this example, it is hard to justify that neighborhoods with a low median income have a high median house value. Therefore we will try enforcing the monotonicity on the median income:

```python
feature_monotones = [1]
```

We then repeat the CV procedure, refit the model and evaluate it on the testset. Below is the result:
```python
Number of boosting rounds 59,       Training RMSE: 0.8189,        Testing RMSE: 0.8279
```

Looks like compared to before the training error slightly increased while testing error slightly decreased. We may have reduced overfitting and improved our performance on the testset. However, given that statistical uncertainties on these numbers are probably just as big as the differences, it is just a hypothesis. For this example, the bottom line is that adding monotonicity constraint does not significantly hurt the performance.

Now we can check the variable dependency again:

![One feature, constraint](https://github.com/XiaoxiaoWang87/xgboost_mono_test/blob/master/w_constraint_one_feature.png)

Great! Now the response is monotonically increasing with the predictor. This model has also become a bit easier to explain.

We can also enforce monotonicity constraints while fitting multiple features. For example:

```python
feature_names = ['MedInc', 'AveOccup', 'HouseAge']
feature_monotones = [1, -1, 1]
```

![Three features](https://github.com/XiaoxiaoWang87/xgboost_mono_test/blob/master/w_constraint_three_feature.png)

We assume that the median house value is positively correlated with median income and house age, but negatively correlated with average house occupancy.

Is it a good idea to enforce monotonicity constraints on features? It depends. For the example here, we didn't see a significant performance decrease, and we think the directions of these variables make intuitive sense. For other cases, especially when the number of variables are large, it may be difficult and even dangerous to do so. It certainly relies on a lot of domain expertise and exploratory analysis to fit a model that is "as simple as possible, but no simpler".

(The iPython Notebook of this tutorial can be found [here](https://github.com/XiaoxiaoWang87/xgboost_mono_test/blob/master/xgboost_monotonicity_tutorial.ipynb))

##Bibliography

[1]: T. Hastie, R. Tibshirani and J. Friedman, “Elements of Statistical Learning Ed. 2”, Springer, 2009, pp. 371-375

[2]: Friedman, Jerome H. "Greedy function approximation: a gradient boosting machine." Annals of statistics (2001): 1189-1232.
