import random

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

from openml.utils.preprocessing import ConditionalImputer


CV_ITT = 3
RS_ITT = 200

def get_naive_bayes(nominal_indices):
    estimator = Pipeline(steps=[('Imputer', ConditionalImputer(strategy='median',
                                                               empty_attribute_constant=0,
                                                               categorical_features=nominal_indices,
                                                               strategy_nominal='most_frequent')),
                                ('OneHotEncoder', OneHotEncoder(sparse=False,
                                                                handle_unknown='ignore',
                                                                categorical_features=nominal_indices)),
                                ('VarianceThreshold', VarianceThreshold()),
                                ('Estimator', GaussianNB())])
    return estimator

def get_decision_tree(nominal_indices):
    min_samples_split_range = [2 ** x for x in range(1, 7 + 1)]
    min_samples_leaf_range = [2 ** x for x in range(0, 6 + 1)]

    param_dist = {'min_samples_split': min_samples_split_range,
                  'min_samples_leaf': min_samples_leaf_range}
    grid_search = GridSearchCV(DecisionTreeClassifier(), param_dist, cv=CV_ITT)

    estimator = Pipeline(steps=[('Imputer', ConditionalImputer(strategy='median',
                                                               empty_attribute_constant=0,
                                                               categorical_features=nominal_indices,
                                                               strategy_nominal='most_frequent')),
                                ('OneHotEncoder', OneHotEncoder(sparse=False,
                                                                handle_unknown='ignore',
                                                                categorical_features=nominal_indices)),
                                ('VarianceThreshold', VarianceThreshold()),
                                ('Estimator', grid_search)])
    return estimator

def get_svm(nominal_indices):
    param_range = [2 ** x for x in range(-12, 12 + 1)]

    param_dist = {'C': param_range,
                  'gamma': param_range}
    random_search = RandomizedSearchCV(SVC(kernel='rbf'), param_dist, cv=CV_ITT, n_iter=RS_ITT)

    estimator = Pipeline(steps=[('Imputer', ConditionalImputer(strategy='median',
                                                               empty_attribute_constant=0,
                                                               categorical_features=nominal_indices,
                                                               strategy_nominal='most_frequent')),
                                ('OneHotEncoder', OneHotEncoder(sparse=False,
                                                                handle_unknown='ignore',
                                                                categorical_features=nominal_indices)),
                                ('VarianceThreshold', VarianceThreshold()),
                                ('Estimator', random_search)])
    return estimator

def get_gradient_boosting(nominal_indices):
    learning_rate_range = [10 ** x for x in range(-4, -1 + 1)]
    max_depth_range = list(range(1, 5 + 1))
    n_estimators_range = list(range(500, 1000 + 1))

    param_dist = {'learning_rate': learning_rate_range,
                  'max_depth': max_depth_range,
                  'n_estimators': n_estimators_range}
    random_search = RandomizedSearchCV(GradientBoostingClassifier(), param_dist, cv=CV_ITT, n_iter=RS_ITT)

    estimator = Pipeline(steps=[('Imputer', ConditionalImputer(strategy='median',
                                                               empty_attribute_constant=0,
                                                               categorical_features=nominal_indices,
                                                               strategy_nominal='most_frequent')),
                                ('OneHotEncoder', OneHotEncoder(sparse=False,
                                                                handle_unknown='ignore',
                                                                categorical_features=nominal_indices)),
                                ('VarianceThreshold', VarianceThreshold()),
                                ('Estimator', random_search)])
    return estimator

def get_knn(nominal_indices):
    n_neighbours_range = list(range(1, 50 + 1))

    param_dist = {'n_neighbors': n_neighbours_range}
    grid_search = GridSearchCV(NearestNeighbors(), param_dist, cv=CV_ITT)

    estimator = Pipeline(steps=[('Imputer', ConditionalImputer(strategy='median',
                                                               empty_attribute_constant=0,
                                                               categorical_features=nominal_indices,
                                                               strategy_nominal='most_frequent')),
                                ('OneHotEncoder', OneHotEncoder(sparse=False,
                                                                handle_unknown='ignore',
                                                                categorical_features=nominal_indices)),
                                ('VarianceThreshold', VarianceThreshold()),
                                ('Estimator', grid_search)])
    return estimator


def get_mlp(nominal_indices):
    hidden_layes_sizes_range = [2 ** x for x in range(5, 11 + 1)] # log scale (!)
    learning_rate_init_range = [10 ** x for x in range(-5, 0 + 1)]
    alpha_range = [10 ** x for x in range(-7, -4 + 1)]
    max_iter_range = [2 ** x for x in range(1, 11 + 1)]
    momentum_range = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    param_dist = {'hidden_layer_sizes': hidden_layes_sizes_range,
                  'learning_rate_init': learning_rate_init_range,
                  'alpha': alpha_range,
                  'max_iter': max_iter_range,
                  'momentum': momentum_range}
    random_search = RandomizedSearchCV(MLPClassifier(), param_dist, cv=CV_ITT, n_iter=RS_ITT)

    estimator = Pipeline(steps=[('Imputer', ConditionalImputer(strategy='median',
                                                               empty_attribute_constant=0,
                                                               categorical_features=nominal_indices,
                                                               strategy_nominal='most_frequent')),
                                ('OneHotEncoder', OneHotEncoder(sparse=False,
                                                                handle_unknown='ignore',
                                                                categorical_features=nominal_indices)),
                                ('VarianceThreshold', VarianceThreshold()),
                                ('Estimator', random_search)])
    return estimator

def get_logistic_regression(nominal_indices):
    C_range = [2 ** x for x in range(-12, 12 + 1)]

    param_dist = {'C': C_range}
    grid_search = GridSearchCV(LogisticRegression(), param_dist, cv=CV_ITT)

    estimator = Pipeline(steps=[('Imputer', ConditionalImputer(strategy='median',
                                                               empty_attribute_constant=0,
                                                               categorical_features=nominal_indices,
                                                               strategy_nominal='most_frequent')),
                                ('OneHotEncoder', OneHotEncoder(sparse=False,
                                                                handle_unknown='ignore',
                                                                categorical_features=nominal_indices)),
                                ('VarianceThreshold', VarianceThreshold()),
                                ('Estimator', grid_search)])
    return estimator

def get_random_forest(nominal_indices):
    max_features_range = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    param_dist = {'max_features': max_features_range}
    grid_search = GridSearchCV(RandomForestClassifier(n_estimators=500), param_dist, cv=CV_ITT)

    estimator = Pipeline(steps=[('Imputer', ConditionalImputer(strategy='median',
                                                               empty_attribute_constant=0,
                                                               categorical_features=nominal_indices,
                                                               strategy_nominal='most_frequent')),
                                ('OneHotEncoder', OneHotEncoder(sparse=False,
                                                                handle_unknown='ignore',
                                                                categorical_features=nominal_indices)),
                                ('VarianceThreshold', VarianceThreshold()),
                                ('Estimator', grid_search)])
    return estimator


def get_random_estimator(nominal_indices):
    estimators = [get_naive_bayes, get_decision_tree, get_logistic_regression, get_gradient_boosting,
                  get_svm, get_random_forest, get_mlp, get_knn]
    estimator = random.choice(estimators)
    return estimator(nominal_indices)
