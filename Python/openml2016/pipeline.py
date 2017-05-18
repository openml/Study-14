import copy

import numpy as np
import scipy.stats

import sklearn.ensemble
import sklearn.feature_selection
import sklearn.linear_model
import sklearn.naive_bayes
import sklearn.neighbors
import sklearn.neural_network
import sklearn.pipeline
import sklearn.preprocessing
import sklearn.svm
import sklearn.tree

from . import preprocessing
from . import distributions

np.random.seed(1)


def _get_pipeline_preprocessing(version='mixed'):
    param_dist = {}

    categorical_preprocessing_pipeline = \
            [['Imputer', sklearn.preprocessing.Imputer(strategy='most_frequent')],
             # sparse data makes Naive Bayes and GBM fail, but otherwise it will
             # take too long fitting huge datasets full of zeros...
             ['OneHotEncoder', sklearn.preprocessing.OneHotEncoder(
                 sparse=True, categorical_features='all', handle_unknown='ignore')]]
    numerical_preprocessing_pipeline = \
            [['Imputer', sklearn.preprocessing.Imputer(strategy='median')]]

    if version == 'mixed':
        categorical_preprocessing_pipeline = \
            sklearn.pipeline.Pipeline(
                [['ItemSelector', preprocessing.ItemSelector()]] +
                categorical_preprocessing_pipeline)
        numerical_preprocessing_pipeline = \
            sklearn.pipeline.Pipeline(
                [['ItemSelector', preprocessing.ItemSelector()]] +
                 numerical_preprocessing_pipeline)

        fu = sklearn.pipeline.FeatureUnion(
            transformer_list=[['categorical', categorical_preprocessing_pipeline],
                              ['numerical', numerical_preprocessing_pipeline]],
            n_jobs=1)
    elif version == 'numerical':
        fu = sklearn.pipeline.Pipeline(numerical_preprocessing_pipeline)
    elif version == 'categorical':
        fu = sklearn.pipeline.Pipeline(categorical_preprocessing_pipeline)
    else:
        raise ValueError('Unknown version', version)

    pipeline = [['Preprocessing', fu],
                ['VarianceThreshold', sklearn.feature_selection.VarianceThreshold()]]

    return param_dist, pipeline


def get_SVM(version):
    param_dist, pipeline = _get_pipeline_preprocessing(version)

    param_dist.update({'classifier__C': distributions.loguniform(
                           base=2, low=2**-12, high=2**12),
                       'classifier__gamma': distributions.loguniform(
                           base=2, low=2**-12, high=2**12)})

    pipeline.append(['scaler', sklearn.preprocessing.StandardScaler()])
    pipeline.append(['classifier', sklearn.svm.SVC(probability=True,
                                                   cache_size=2000)])

    pipeline = sklearn.pipeline.Pipeline(pipeline)
    return param_dist, pipeline


def get_decision_tree(version):
    param_dist, pipeline = _get_pipeline_preprocessing(version)

    param_dist.update({'classifier__min_samples_split':
                           distributions.loguniform_int(
                               base=2, low=2**1, high=2**7),
                       'classifier__min_samples_leaf':
                           distributions.loguniform_int(
                               base=2, low=2**0, high=2**6)})

    pipeline.append(['classifier', sklearn.tree.DecisionTreeClassifier()])

    pipeline = sklearn.pipeline.Pipeline(pipeline)
    return param_dist, pipeline


def get_gradient_boosting(version):
    param_dist, pipeline =  _get_pipeline_preprocessing(version)

    param_dist.update({'classifier__learning_rate':
                           distributions.loguniform(
                               base=10, low=10e-4, high=10e-1),
                       'classifier__max_depth': scipy.stats.randint(1, 6),
                       'classifier__n_estimators':
                           scipy.stats.randint(500, 10001)})

    pipeline.append(['classifier',
                     sklearn.ensemble.GradientBoostingClassifier(
                         n_estimators=100)])

    pipeline = sklearn.pipeline.Pipeline(pipeline)
    return param_dist, pipeline


def get_kNN(version):
    param_dist, pipeline = _get_pipeline_preprocessing(version)

    param_dist.update({'classifier__n_neighbors': scipy.stats.randint(1, 51)})

    pipeline.append(['scaler', sklearn.preprocessing.StandardScaler()])
    pipeline.append(['classifier', sklearn.neighbors.KNeighborsClassifier()])

    pipeline = sklearn.pipeline.Pipeline(pipeline)
    return param_dist, pipeline


def get_naive_bayes(version):
    param_dist, pipeline = _get_pipeline_preprocessing(version)

    pipeline.append(['scaler', sklearn.preprocessing.StandardScaler()])
    pipeline.append(['classifier', sklearn.naive_bayes.GaussianNB()])

    pipeline = sklearn.pipeline.Pipeline(pipeline)
    return param_dist, pipeline


def get_neural_network(version):
    param_dist, pipeline = _get_pipeline_preprocessing(version)

    param_dist_new = {
        'classifier__learning_rate_init': distributions.loguniform(
            base=10, low=10e-6, high=10e-1),
        'classifier__alpha': distributions.loguniform(
            base=10, low=10e-7, high=10e-4),
        'classifier__max_iter': scipy.stats.randint(2, 1001),
        # uniform distribution between 0.1 and 0.9
        'classifier__momentum': scipy.stats.uniform(loc=0.1, scale=0.8)}

    param_dist_1 = copy.deepcopy(param_dist)
    param_dist_1.update(param_dist_new)
    param_dist_1['classifier__hidden_layer_sizes'] = (scipy.stats.randint(32, 2049),)
    param_dist_2 = copy.deepcopy(param_dist)
    param_dist_2.update(param_dist_new)
    param_dist_2['classifier__hidden_layer_sizes'] = \
        (scipy.stats.randint(32, 2049), scipy.stats.randint(32, 2049))

    pipeline.append(['scaler', sklearn.preprocessing.StandardScaler()])
    pipeline.append(['classifier', sklearn.neural_network.MLPClassifier(
        activation='tanh', solver='adam')])

    pipeline = sklearn.pipeline.Pipeline(pipeline)
    return param_dist, pipeline


def get_logistic_regression(version):
    param_dist, pipeline = _get_pipeline_preprocessing(version)

    # TODO check if C==alpha?
    param_dist.update({'classifier__C': distributions.loguniform(
                           base=2, low=2**-12, high=2**12)})

    pipeline.append(['scaler', sklearn.preprocessing.StandardScaler()])
    pipeline.append(['classifier', sklearn.linear_model.LogisticRegression(
        solver='lbfgs', multi_class='multinomial')])

    pipeline = sklearn.pipeline.Pipeline(pipeline)
    return param_dist, pipeline


def get_random_forest(version, num_features):
    param_dist, pipeline = _get_pipeline_preprocessing(version)
    param_dist.update({'classifier__max_features':
                       distributions.loguniform(
                           base=num_features, low=num_features**0.1,
                           high=num_features**0.9)})

    pipeline.append(['classifier',
                     sklearn.ensemble.RandomForestClassifier(n_estimators=500)])

    pipeline = sklearn.pipeline.Pipeline(pipeline)
    return param_dist, pipeline




