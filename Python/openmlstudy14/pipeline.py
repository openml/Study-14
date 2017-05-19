import os
import random
import sys

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import OneHotEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

this_directory = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(this_directory, '..'))

from openmlstudy14.distributions import loguniform, loguniform_int
from openmlstudy14.preprocessing import ConditionalImputer

scoring='accuracy'
error_score=0.0
CV_ITT=3
RS_ITT=20
grid_arguments = dict(scoring='accuracy',
                     error_score=error_score,
                     cv=CV_ITT)
rs_arguments = dict(scoring='accuracy',
                    error_score=error_score,
                    cv=CV_ITT,
                    n_iter=RS_ITT)


def _get_pipeline(nominal_indices, sklearn_model):
    steps = [('Imputer', ConditionalImputer(strategy='median',
                                           fill_empty=0,
                                           categorical_features=nominal_indices,
                                           strategy_nominal='most_frequent')),
             ('OneHotEncoder', OneHotEncoder(sparse=False,
                                            handle_unknown='ignore',
                                            categorical_features=nominal_indices)),
             ('VarianceThreshold', VarianceThreshold()),
             ('Estimator', sklearn_model)]
    return Pipeline(steps=steps)


def get_naive_bayes(nominal_indices):
    return _get_pipeline(nominal_indices, GaussianNB)


def get_decision_tree(nominal_indices):
    param_dist = {'Estimator__min_samples_split': loguniform_int(base=2, low=2**1, high=2**7),
                  'Estimator__min_samples_leaf': loguniform_int(base=2, low=2**0, high=2**6)}
    decision_tree = _get_pipeline(nominal_indices, DecisionTreeClassifier())
    random_search = RandomizedSearchCV(decision_tree,
                                       param_dist,
                                       **rs_arguments)
    return random_search


def get_svm(nominal_indices):
    param_dist = {'Estimator__C': loguniform(base=2, low=2**-12, high=2**12),
                  'Estimator__gamma': loguniform(base=2, low=2**-12, high=2**12)}
    svm = _get_pipeline(nominal_indices, SVC(kernel='rbf',probability=True))
    random_search = RandomizedSearchCV(svm,
                                       param_dist,
                                       **rs_arguments)
    return random_search


def get_gradient_boosting(nominal_indices):
    param_dist = {'Estimator__learning_rate': loguniform(base=10, low=2**-4, high=2**-1),
                  'Estimator__max_depth': list(range(1, 5 + 1)),
                  'Estimator__n_estimators': list(range(500, 1000 + 1))}
    boosting = _get_pipeline(nominal_indices, GradientBoostingClassifier())
    random_search = RandomizedSearchCV(boosting,
                                       param_dist,
                                       **rs_arguments)
    return random_search


def get_knn(nominal_indices):

    param_dist = {'Estimator__n_neighbors': list(range(1, 50 + 1))}
    knn = _get_pipeline(nominal_indices, NearestNeighbors())
    grid_search = GridSearchCV(knn, param_dist, **grid_arguments)
    return grid_search


def get_mlp(nominal_indices):
    param_dist = {'Estimator__hidden_layer_sizes': loguniform_int(base=2, low=2**5, high=2**11),
                  'Estimator__learning_rate_init': loguniform(base=10, low=2**-5, high=2**0),
                  'Estimator__alpha': loguniform(base=10, low=2**-7, high=2**-4),
                  'Estimator__max_iter': loguniform_int(base=2, low=2**1, high=2**11),
                  'Estimator__momentum': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}
    mlp = _get_pipeline(nominal_indices, MLPClassifier())
    random_search = RandomizedSearchCV(mlp, param_dist, **rs_arguments)
    return random_search

def get_logistic_regression(nominal_indices):
    param_dist = {'Estimator__C': loguniform(base=2, low=2**-12, high=2**12)}
    logreg = _get_pipeline(nominal_indices, LogisticRegression())
    random_search = RandomizedSearchCV(logreg, param_dist, **rs_arguments)
    return random_search

def get_random_forest(nominal_indices):
    param_dist = {'Estimator__max_features': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}
    randomforest = _get_pipeline(nominal_indices, RandomForestClassifier(n_estimators=500))
    grid_search = GridSearchCV(randomforest, param_dist, **grid_arguments)
    return grid_search


def get_random_estimator(nominal_indices):
    estimators = [get_naive_bayes, get_decision_tree, get_logistic_regression, get_gradient_boosting,
                  get_svm, get_random_forest, get_mlp, get_knn]
    estimator = random.choice(estimators)
    print("invoked estimator with %s" %str(estimator))
    return estimator(nominal_indices)
